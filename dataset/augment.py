# https://github.com/ebouteillon/freesound-audio-tagging-2019/blob/master/code/training-cnn-model1.ipynb
import torch
import numpy as np


def spec_augment(last_input, last_target, masking_max_percentage=0.25):
    # Spec Augmentation
    shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
    x1, y1 = last_input[shuffle], last_target[shuffle]

    batch_size, channels, height, width = last_input.size()
    h_percentage = np.random.uniform(low=0., high=masking_max_percentage, size=batch_size)
    w_percentage = np.random.uniform(low=0., high=masking_max_percentage, size=batch_size)
    alpha = (h_percentage + w_percentage) - (h_percentage * w_percentage)
    alpha = last_input.new(alpha)
    alpha = alpha.unsqueeze(1)

    new_input = last_input.clone()

    for i in range(batch_size):
        h_mask = int(h_percentage[i] * height)
        h = int(np.random.uniform(0.0, height - h_mask))
        new_input[i, :, h:h + h_mask, :] = x1[i, :, h:h + h_mask, :]

        w_mask = int(w_percentage[i] * width)
        w = int(np.random.uniform(0.0, width - w_mask))
        new_input[i, :, :, w:w + w_mask] = x1[i, :, :, w:w + w_mask]

    new_target = (1 - alpha) * last_target + alpha * y1
    return new_input, new_target


# Inspired from fastai implementation of https://arxiv.org/abs/1710.09412
def mixup(last_input, last_target, alpha=0.4):
    lambd = np.random.beta(alpha, alpha, last_target.size(0))
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lambd = last_input.new(lambd)
    shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
    x1, y1 = last_input[shuffle], last_target[shuffle]
    new_input = (last_input * lambd.view(lambd.size(0), 1, 1, 1) + x1 * (1 - lambd).view(lambd.size(0), 1, 1, 1))
    if len(last_target.shape) == 2:
        lambd = lambd.unsqueeze(1).float()
    new_target = last_target.float() * lambd + y1.float() * (1 - lambd)
    return new_input, new_target