import torch


def weight_quant_fn(weight, num_bits, num_std=2):
    mean, std = weight.mean(), weight.std()
    clip_val = (mean - num_std * std, mean + num_std * std)
    clip_val = torch.tensor(list(clip_val))

    return quant_uniform(weight,num_bits,clip_val)


def quant_uniform(input, num_bits=2, clip_val=None):
    if clip_val!=None:
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
    print(f"uniform quant with {num_bits} bits")
    alpha = (input.max() - input.min()).detach()
    beta = input.min().detach()
    input_normalized = (input - beta) / (alpha + 1e-8)
    s = (2 ** num_bits - 1)
    quant_input = torch.round(input_normalized * s).div(s)
    output = quant_input * (alpha + 1e-8) + beta
    return output
