import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# This implementation is an attempt to adopt Nicholas Carlini's implementation
# for the CW attack (from https://arxiv.org/abs/1608.04644).
# You find the original code here: https://github.com/carlini/nn_robust_attacks/blob/master/li_attack.py


class CWAttack(nn.Module):
    def __init__(
        self,
        normball,
        init_const=1e-3,
        decrease_factor=0.9,
        learning_rate=1e-2,
        iterations=1000,
    ):
        super(CWAttack, self).__init__()
        self.const = init_const
        self.decrease_factor = decrease_factor
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.normball = normball

    def forward(self, x, y, model, loss_fn):
        # In the original CW attack (designed for traditional classifiers),
        # each iteration the can terminate early if the class prediction changes
        # since the class prediction is much more costly for DML, we were forced
        # to omit this ability.

        max_x_adv = torch.zeros_like(x).to(x.device)
        max_loss = torch.zeros(x.shape[0]).to(x.device)

        const = self.const

        tau = 1.0
        ntau = self.normball.epsilon * 2.0
        perb = self.normball.sample_like(x)

        optimizer = optim.Adam([perb], lr=self.learning_rate)
        perb.requires_grad = True
        while tau > 1.0 / 256:
            if ntau < self.normball.epsilon:
                break
            while const < 20:
                for _ in range(self.iterations):
                    with torch.enable_grad():
                        x_adv = x + perb
                        output = model(x_adv)
                        loss1 = loss_fn(output, y)
                        exceed = torch.abs(perb) - tau
                        loss2 = torch.sum(
                            torch.max(torch.zeros_like(exceed), exceed)
                        )
                        loss = -const * loss1 + loss2
                        optimizer.zero_grad()
                        loss.backward()
                        # yields more effective attacks from experimentation
                        perb.grad = self.normball.step(x_adv, perb.grad)
                        optimizer.step()

                const *= 2.0
            ntau = torch.max(torch.abs(perb))
            if ntau < tau:
                tau = ntau
            tau *= self.decrease_factor

            # small adoption that ensure we only retain the best
            # performing perturbation
            x_adv = perb + x
            x_adv = self.normball.clip(x_adv, x)
            all_loss = loss_fn(model(x_adv), y).detach()
            higher_loss = all_loss >= max_loss
            max_x_adv[higher_loss] = x_adv.detach()[higher_loss]

        return max_x_adv.detach()
