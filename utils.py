import torch
import math
import random
from torch import nn
from utils_quant import weight_quant_fn
import torch.nn.functional as F


def explore_grad(weight):
    if weight.requires_grad is True:
        print(weight.shape)
        print(weight)
        print(weight.grad)


def low_rank_decomposition(weight, reduced_rank=0.15):
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"

    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    reduced_rank = int(reduced_rank)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return L, R


# 量化和低秩分解的初始迭代函数
def quant_first_iter(weight, L, R, reduced_rank, int_bit, **kwargs):
    low_rank_product = L @ R if torch.is_tensor(L) else 0
    residual = weight - low_rank_product  # 计算残差
    quant_w = weight_quant_fn(residual, num_bits=int_bit)
    output = low_rank_decomposition(weight - quant_w, reduced_rank=reduced_rank)
    L, R = output[0], output[1]
    final_residual = weight - quant_w - L @ R  # 计算最终残差
    return weight, L, R, quant_w, final_residual  # 返回更新后的权重、低秩矩阵 L 和 R、量化后的权重以及最终残差


def replace_module(module, allow_name=None, block_name=None, reduced_rank=32, int_bit=2, args=None, **kwargs):

    if allow_name is None:
        allow_name = ['query', 'key', 'value', 'dense', 'attention']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Linear and any(attr_str in an for an in allow_name):
            if args.lora:
                weights = target_attr.weight
                L, R = 0, 0
                linear_loras = LinearQuantLoRA(target_attr.in_features, target_attr.out_features, reduced_rank=int(reduced_rank),
                                               has_bias=True,  args=args)
                linear_loras.initialize_weight(weights, L, R, 0, target_attr.bias)
            elif args.qlora:
                weights = target_attr.weight
                L, R = 0, 0
                quant_w = weight_quant_fn(weights, num_bits=int_bit)
                linear_loras = LinearQuantLoRA(target_attr.in_features, target_attr.out_features, reduced_rank=int(reduced_rank),
                                               has_bias=True,  args=args)
                linear_loras.initialize_weight(quant_w, L, R, 0, target_attr.bias)
            elif args.loftq:
                weights = target_attr.weight
                L, R = 0, 0
                for i in range(args.num_iter):
                    weights, L, R, quant_w, final_residual = quant_first_iter(weights, L, R, reduced_rank, int_bit, **kwargs)
                linear_loras = LinearQuantLoRA(target_attr.in_features, target_attr.out_features, reduced_rank=int(reduced_rank),
                                               has_bias=True,  args=args)
                linear_loras.initialize_weight(quant_w, L, R, 0, target_attr.bias)
            elif args.oqfv:
                weights = target_attr.weight
                L, R = 0, 0
                weights, L, R, quant_w, final_residual = quant_first_iter(weights, L, R, reduced_rank,int_bit, **kwargs)
                linear_loras = LinearQuantLoRA(target_attr.in_features, target_attr.out_features,
                                               reduced_rank=int(reduced_rank),
                                               has_bias=True, args=args)
                linear_loras.initialize_weight(quant_w, L, R, 0, target_attr.bias)

            setattr(module, attr_str, linear_loras)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            replace_module(immediate_child_module, allow_name, block_name, reduced_rank, int_bit, args=args, **kwargs)


def show_model_stats(model, mark_only_lora_as_trainable=True):
    total = 0
    lr_adapter = 0
    if mark_only_lora_as_trainable:
        for n, m in model.vit.named_parameters():
            if 'lora' in n or 'left' in n or 'right' in n or 'down' in n or 'up' in n:
                m.requires_grad = True
                lr_adapter += m.numel()
            else:
                m.requires_grad = False
            print(n, m.shape, m.requires_grad)
            total += m.numel()

    else:
        for n, m in model.vit.named_parameters():
            if "quant" in n or "word_embeddings.weight" in n:
                print(n, m)
            if m.requires_grad:
                lr_adapter += m.numel()
                print(lr_adapter)
            total += m.numel()
    print(f"Total trainable parameters {lr_adapter}")
    print(f"We finetune about {lr_adapter / total} ratio of percentages")


class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LinearQuantLoRA(nn.Module):
    def __init__(self, in_feature, out_feature, reduced_rank, has_bias=True, args=None):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.reduced_rank = reduced_rank
        self.has_bias = has_bias
        self.quant = nn.Linear(in_feature, out_feature, bias=False)

        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(out_feature, requires_grad=True))

        self.has_lora_adapter = args.lora
        self.has_qlora_adapter = args.qlora
        self.has_loftq_adapter = args.loftq
        self.has_oqfv_adapter = args.oqfv

        if self.has_lora_adapter:
            print(f"low rank adapter with rank {reduced_rank} using LoRa")
            self.lora_A = nn.Linear(in_feature, reduced_rank, bias=False)
            self.lora_B = nn.Linear(reduced_rank, out_feature, bias=False)
        if self.has_qlora_adapter:
            print(f"low rank adapter with rank {reduced_rank} using QLoRa")
            self.lora_A = nn.Linear(in_feature, reduced_rank, bias=False)
            self.lora_B = nn.Linear(reduced_rank, out_feature, bias=False)
        if self.has_loftq_adapter:
            print(f"low rank adapter with rank {reduced_rank} using LoftQ")
            self.left = nn.Linear(in_feature, reduced_rank, bias=False)
            self.right = nn.Linear(reduced_rank, out_feature, bias=False)
        if self.has_oqfv_adapter:
            print(f"low rank adapter with rank {reduced_rank} using OQFV")
            self.lora_A = nn.Linear(in_feature, reduced_rank, bias=False)
            self.lora_B = nn.Linear(reduced_rank, out_feature, bias=False)

    def initialize_weight(self, quant_weight, left_weight, right_weight, sparse_weight=None, bias=None):
        self.quant.weight = nn.Parameter(quant_weight, requires_grad=False)  # Freeze the backbone
        if self.has_bias:
            self.bias = nn.Parameter(bias, requires_grad=True)

        if self.has_lora_adapter:
            lora_A_weight = nn.Parameter(self.quant.weight.new_zeros((self.reduced_rank, self.in_feature)), requires_grad=True)
            lora_B_weight = nn.Parameter(self.quant.weight.new_zeros((self.out_feature, self.reduced_rank), requires_grad=True))
            nn.init.kaiming_uniform_(lora_A_weight, a=math.sqrt(5))
            nn.init.zeros_(lora_B_weight)
            self.lora_A.weight = lora_A_weight
            self.lora_B.weight = lora_B_weight
        if self.has_qlora_adapter:
            lora_A_weight = nn.Parameter(self.quant.weight.new_zeros((self.reduced_rank, self.in_feature)), requires_grad=True)
            lora_B_weight = nn.Parameter(self.quant.weight.new_zeros((self.out_feature, self.reduced_rank), requires_grad=True))
            nn.init.kaiming_uniform_(lora_A_weight, a=math.sqrt(5))
            nn.init.zeros_(lora_B_weight)
            self.lora_A.weight = lora_A_weight
            self.lora_B.weight = lora_B_weight
        if self.has_loftq_adapter:
            self.left.weight = nn.Parameter(left_weight, requires_grad=True)
            self.right.weight = nn.Parameter(right_weight, requires_grad=True)
        if self.has_eqft_adapter:
            self.lora_A.weight = nn.Parameter(right_weight, requires_grad=True)
            self.lora_B.weight = nn.Parameter(left_weight, requires_grad=True)
        if self.has_oqfv_adapter:
            self.lora_A.weight = nn.Parameter(right_weight, requires_grad=True)
            self.lora_B.weight = nn.Parameter(left_weight, requires_grad=True)

    def forward(self, x):
        HX = self.quant(x)
        if self.has_lora_adapter:
            lora_A_output = self.lora_A(x)
            ABX = self.lora_B(lora_A_output)
            Y = HX + ABX + self.bias if self.has_bias else HX + ABX
        if self.has_qlora_adapter:
            lora_A_output = self.lora_A(x)
            ABX = self.lora_B(lora_A_output)
            Y = HX + ABX + self.bias if self.has_bias else HX + ABX
        if self.has_loftq_adapter:
            right_output = self.right(x)
            LRX = self.left(right_output)
            Y = HX + LRX + self.bias if self.has_bias else HX + LRX
        if self.has_oqfv_adapter:
            lora_A_output = self.lora_A(x)
            ABX = self.lora_B(lora_A_output)
            Y = HX + ABX + self.bias if self.has_bias else HX + ABX
        return Y
