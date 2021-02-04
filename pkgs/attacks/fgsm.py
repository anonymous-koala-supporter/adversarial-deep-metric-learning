import torch
import torch.nn as nn


class FGSMAttack(nn.Module):
    def __init__(self, normball, alpha=None, random_init=False):
        super(FGSMAttack, self).__init__()
        self.alpha = (
            normball.epsilon * 1.25 if alpha is None else alpha
        )  # default: from "Fast is better than free"-paper
        self.normball = normball
        self.random_init = random_init

    def forward(self, x, y, model, loss_fn):
        x = x.detach()
        delta = torch.zeros_like(x)

        if self.random_init:
            delta += self.normball.sample_like(x)

        delta.requires_grad = True

        with torch.enable_grad():
            output = model(x + delta)
            loss = loss_fn(output, y)
            loss.backward()

        grads = delta.grad.detach()
        x_adv = x + self.normball.step(x, grads) * self.alpha
        x_adv = self.normball.clip(x_adv, x)

        return x_adv
