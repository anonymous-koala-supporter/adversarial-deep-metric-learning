import torch
import torch.nn as nn


class PGDAttack(nn.Module):
    def __init__(
        self, normball, iterations=5, alpha=None, random_restarts=1,
    ):
        super(PGDAttack, self).__init__()
        self.alpha = (
            alpha if alpha is not None else (2 * normball.epsilon) / iterations
        )
        self.iterations = iterations
        self.random_restarts = random_restarts
        self.normball = normball

    def forward(self, x, y, model, loss_fn):
        x = x.detach()

        max_loss = torch.ones(x.shape[0]).to(x.device) * float("-inf")
        max_adv = torch.zeros_like(x).to(x.device)

        for _ in range(self.random_restarts):
            x_adv = x + self.normball.sample_like(x)
            x_adv.requires_grad = True
            opt = torch.optim.SGD([x_adv], lr=self.alpha)

            for _ in range(self.iterations):
                with torch.enable_grad():
                    opt.zero_grad()
                    output = model(x_adv)
                    loss = -loss_fn(
                        output, y
                    )  # optimizers minimize, thus the loss has to be inverted for maximization
                    loss.backward()
                    x_adv.grad = self.normball.step(x_adv, x_adv.grad)
                    opt.step()

                x_adv.data = self.normball.clip(x_adv, x)

            with torch.no_grad():
                all_loss = loss_fn(model(x_adv), y, reduction="none")
                higher_loss = all_loss >= max_loss
                max_adv[higher_loss] = x_adv[higher_loss]

            max_loss = torch.max(max_loss, all_loss)

        return max_adv
