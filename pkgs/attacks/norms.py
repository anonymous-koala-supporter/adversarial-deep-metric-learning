import numpy

import torch


class NormBall(torch.nn.Module):
    def __init__(self, epsilon, global_min=0.0, global_max=1.0):
        super(NormBall, self).__init__()
        self.epsilon = torch.tensor(epsilon)
        self.global_min = global_min
        self.global_max = global_max

    def sample_like(self, x):
        return self.sample_shape(x.shape).to(x.device)

    def sample_shape(self, shape):
        raise NotImplementedError

    def clip(self, x, x_ref, global_min=0, global_max=1):
        delta = x - x_ref
        new_delta = self.clip_to_epsilon(delta)
        x.data += new_delta - delta
        x = torch.clamp(x, min=global_min, max=global_max)

        return x

    def clip_to_epsilon(self, delta):
        raise NotImplementedError

    def step(self, x_adv, grads):
        raise NotImplementedError


class L2NormBall(NormBall):
    def __init__(self, epsilon):
        super(L2NormBall, self).__init__(epsilon)

    def sample_shape(self, shape):
        z = torch.randn(shape).reshape(shape[0], -1)
        r = z.norm(p=2, dim=1).unsqueeze(1)
        return (z / r).reshape(shape) * self.epsilon

    def clip_to_epsilon(self, delta):
        _s = delta.shape

        delta = delta.reshape(_s[0], -1)
        delta_norm = delta.norm(p=2, dim=1).unsqueeze(1)
        delta = (
            delta
            / delta_norm.clamp(min=1e-12)
            * delta_norm.clamp(max=self.epsilon)
        ).reshape(_s)

        return delta

    def step(self, x_adv, grads):
        bad_pos = ((x_adv == self.global_max) & (grads > 0)) | (
            (x_adv == self.global_min) & (grads < 0)
        )
        grads[bad_pos] = 0

        grads = grads.reshape(len(x_adv), -1)
        grads /= grads.norm(p=2, dim=1).clamp(max=1e-12).unsqueeze(1)
        grads = grads.reshape(x_adv.shape)

        return grads.sign()


class LinfNormBall(NormBall):
    def __init__(self, epsilon):
        super(LinfNormBall, self).__init__(epsilon)

    def sample_shape(self, shape):
        return torch.zeros(shape).uniform_(-self.epsilon, self.epsilon)

    def clip_to_epsilon(self, delta):
        return torch.clamp(delta, max=self.epsilon, min=-self.epsilon)

    def step(self, x_adv, grads):
        return grads.sign()
