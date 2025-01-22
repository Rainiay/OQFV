import torch.nn as nn
import torch.optim as optim
import argparse
import logging

from tqdm.auto import tqdm

from transformers import (
    ViTForImageClassification,
    SchedulerType,
    get_scheduler,
)
import utils
from avalanche.evaluation.metrics.accuracy import Accuracy
from vtab import *

device = torch.device(0)
DEBUG = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_from_dataloader_random(dataloader, num_samples):
    # 将所有数据加载到内存中
    all_inputs = []
    for inputs, labels in dataloader:
        all_inputs.append(inputs)

    # 将数据展平成一个列表
    all_inputs = torch.cat(all_inputs)

    # 随机选择索引
    indices = random.sample(range(len(all_inputs)), num_samples)

    # 根据索引采样数据
    sampled_inputs = all_inputs[indices]

    return sampled_inputs


# 分布采样
def sample_from_dataloader_distribution(dataloader, num_samples, num_strata=10):
    all_inputs = []
    all_labels = []

    for inputs, labels in dataloader:
        all_inputs.append(inputs)
        all_labels.append(labels)

    all_inputs = torch.cat(all_inputs)
    all_labels = torch.cat(all_labels)

    strata = [all_inputs[all_labels == i] for i in range(num_strata)]

    sampled_inputs = []
    remaining_samples = num_samples

    # 初步采样
    for stratum in strata:
        if len(stratum) > 0:
            num_samples_per_stratum = remaining_samples // num_strata
            indices = random.sample(range(len(stratum)), min(num_samples_per_stratum, len(stratum)))
            sampled_inputs.append(stratum[indices])
            remaining_samples -= len(indices)
        num_strata -= 1

    if remaining_samples > 0 and num_strata > 0:
        for _ in range(remaining_samples):
            selected_stratum = random.choice([stratum for stratum in strata if len(stratum) > 0])
            index = random.choice(range(len(selected_stratum)))
            sampled_inputs.append(selected_stratum[index].unsqueeze(0))

    sampled_inputs = torch.cat(sampled_inputs)
    return sampled_inputs


def optimize_lora_weights(model, calibration, target, learning_rate=1e-3):
    model.train()
    output = model(calibration).logits
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    criterion = nn.MSELoss()

    optimizer.zero_grad()

    loss = criterion(output, target)
    loss.backward(retain_graph=False)

    optimizer.step()

    del target, output, loss, calibration
    torch.cuda.empty_cache()


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    # pbar = tqdm(dl)
    model = model.to(device)
    for batch in dl:  # pbar:
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x).logits
        acc.update(out.argmax(dim=1).view(-1), y)

    return acc.result()


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.")
    parser.add_argument(
        "--sample_method",
        type=str,
        choices=["random", "distribution"],
        default="distribution", help="Sampling method for the calibration dataset.")
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=128,
        help="Batch size for the calibration dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['svhn', 'sun397', 'smallnorb_ele', 'smallnorb_azi',
                'resisc45', 'patch_camelyon', 'oxford_iiit_pet', 'oxford_flowers102',
                'kitti', 'eurosat', 'dtd', 'dsprites_ori', 'dsprites_loc', 'dmlab',
                'diabetic_retinopathy', 'clevr_dist', 'clevr_count', 'cifar', 'caltech101'],
        default='svhn', help="A dataset for fine-tuning model.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

    #############################
    #    Experiment Argument    #
    #############################
    parser.add_argument("--reduced_rank", type=float,  default=32, help="rank of low rank adapters")
    parser.add_argument("--int_bit", type=int,  default=2, help="integer bit")
    parser.add_argument("--lora", action="store_true", help="use lora to init low rank adapters")
    parser.add_argument("--qlora", action="store_true", help="use qlora to init low rank adapters")
    parser.add_argument("--loftq", action="store_true", help="use loftq to init low rank adapters")
    parser.add_argument("--oqfv", action="store_true", help="use oqfv to init low rank adapters")
    parser.add_argument("--num_iter", type=int, default="5", help="The number of iterations")

    args = parser.parse_args()

    print(args)
    return args


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    # load_dataset
    train_dl, test_dl = get_data(args.dataset, train_batch_size=args.train_batch_size,
                                 eval_batch_size=args.eval_batch_size)

    num_classes = get_classes_num(args.dataset)

    model = ViTForImageClassification.from_pretrained(args.model_name_or_path, num_labels=num_classes, ignore_mismatched_sizes=True)

    allow_name = ['query', 'key', 'value',
                  'q_proj', 'k_proj', 'v_proj',
                  'query_proj', 'key_proj', 'value_proj',
                  'out_proj', 'dense', 'attention', 'fc1', 'fc2', 'dense']
    block_name = ['pooler', 'classifier', 'LayerNorm']
    model = model.to(device)

    if args.oqfv is True:
        if args.sample_method == "random":
            calibration = sample_from_dataloader_random(train_dl, args.sample_batch_size)
        if args.sample_method == "distribution":
            calibration = sample_from_dataloader_distribution(train_dl, args.sample_batch_size, num_strata=num_classes)
        calibration = calibration.to(device)
        with torch.no_grad():
            target = model(calibration).logits
        utils.replace_module(model, allow_name=allow_name, block_name=block_name,
                             reduced_rank=args.reduced_rank, quant_method=args.quant_method, int_bit=args.int_bit,
                             args=args)

        utils.show_model_stats(model, mark_only_lora_as_trainable=True)
        optimize_lora_weights(model, calibration.to(device), target, learning_rate=1e-3)
    else:
        utils.replace_module(model, allow_name=allow_name, block_name=block_name,
                             reduced_rank=args.reduced_rank, quant_method=args.quant_method, int_bit=args.int_bit,
                             args=args)

        utils.show_model_stats(model, mark_only_lora_as_trainable=True)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    args.max_train_steps = args.num_train_epochs * len(train_dl)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dl)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dl):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

            if completed_steps % 100 == 0:
                print(f"The current loss is {loss}")
    model.eval()
    evaluation = test(model, test_dl)

    method_flags = {
        "LoRA": args.lora,
        "QLoRA": args.qlora,
        "LoftQ": args.loftq,
        "OQFV": args.oqfv,
    }

    enabled_methods = [name for name, enabled in method_flags.items() if enabled]

    methods = ", ".join(enabled_methods) if enabled_methods else "None"

    print(
        f"dataset {args.dataset} seed {args.seed} method {methods} "
        + f"accuracy: {evaluation}"
    )


if __name__ == "__main__":
    main()

