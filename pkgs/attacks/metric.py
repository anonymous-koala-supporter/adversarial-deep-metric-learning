import torch
import torch.nn as nn
from pkgs.losses import ReferenceDistanceLoss
from pkgs.utils import indices_nn, nearest_k_embeddings


class GlobalAttack(nn.Module):
    def __init__(self, first_order_attack):
        super(GlobalAttack, self).__init__()

        self.first_order_attack = first_order_attack

    def forward(self, index, X, idxs_X, labels, model):
        with torch.no_grad():
            embeddings = model(X)

            nn_idxs = indices_nn(index, embeddings, idxs_X)
            nn_labels = index.labels[nn_idxs].to(labels.device)

            # select only data points for which their nearest neigbor
            # embedding is of the same class
            attack_targets = nn_labels == labels

            if not attack_targets.any():
                return X

            nn_pos_embs = nearest_k_embeddings(
                index,
                embeddings[attack_targets],
                labels[attack_targets],
                kind="pos",
                return_centroids=True,
                k=1,
            )

        X_adv = X.detach().clone()

        loss_fn = ReferenceDistanceLoss(pos_embs=nn_pos_embs)

        X_adv[attack_targets] = self.first_order_attack(
            X_adv[attack_targets], labels[attack_targets], model, loss_fn,
        )

        return X_adv
