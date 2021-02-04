import torch
from pytorch_metric_learning.distances import LpDistance


class ReferenceDistanceLoss(torch.nn.Module):
    def __init__(self, pos_embs=None, neg_embs=None, **kwargs):
        super().__init__(**kwargs)
        self.pos_embs = (
            pos_embs.detach().clone() if pos_embs is not None else None
        )
        self.neg_embs = (
            neg_embs.detach().clone() if neg_embs is not None else None
        )
        self.distance = LpDistance(p=2)

    def forward(self, embeddings, *args, reduction="mean"):
        if len(embeddings) == 0:
            return self.zero_losses()

        pos_loss = torch.tensor(0)
        if self.pos_embs is not None:
            pos_loss = self.distance.pairwise_distance(
                embeddings, self.pos_embs
            )

        neg_loss = torch.tensor(0)
        if self.neg_embs is not None:
            neg_loss = self.distance.pairwise_distance(
                embeddings, self.neg_embs
            )

        loss = pos_loss - neg_loss

        if reduction == "mean":
            return loss.mean()

        if reduction == "none":
            return loss

        raise ValueError(f"unknown reduction: {reduction}")
