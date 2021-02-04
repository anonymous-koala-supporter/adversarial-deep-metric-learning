import numpy as np

import torch


def freeze_model(model):
    model_req_grads = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None

    def unfreeze():
        for i, p in enumerate(model.parameters()):
            p.requires_grad = model_req_grads[i]

    return unfreeze


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
