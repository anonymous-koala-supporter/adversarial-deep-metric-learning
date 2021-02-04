import torch


def perturbation_targets(
    X, labels, miner_output, perturb_rate, perturb_target="pos"
):
    assert len(miner_output) == 3
    per_idx, mine_order = {
        "anc": (0, [0, 1, 2]),
        "pos": (1, [0, 1, 0]),
        "neg": (2, [0, 0, 2]),
    }[perturb_target]
    n = len(miner_output[per_idx])

    adv_idx = (torch.rand(n) <= perturb_rate).bool()
    adv_X_idx = miner_output[per_idx][adv_idx]

    new_adv_idx = (torch.arange(adv_idx.sum()) + len(X)).to(
        miner_output[0].device
    )
    adv_miner_output = [
        miner_output[i][adv_idx].detach().clone() for i in mine_order
    ]
    adv_miner_output[per_idx] = new_adv_idx
    adv_miner_output = tuple(adv_miner_output)

    new_miner_output = [
        miner_output[i].detach().clone() for i in range(len(miner_output))
    ]
    new_miner_output[per_idx][adv_idx] = new_adv_idx
    new_miner_output = tuple(new_miner_output)

    new_X = torch.cat([X, X[adv_X_idx].detach().clone()])
    new_labels = torch.cat([labels, labels[adv_X_idx].detach()])

    return new_X, new_labels, adv_miner_output, new_miner_output


def max_per_anchor(miner_output, n_anchors):
    assert len(miner_output) == 3
    a, p, n = miner_output

    idx = torch.cat(
        [
            torch.multinomial((a == c).float(), n_anchors)
            for c in torch.unique(a)
        ]
    )

    return (a[idx], p[idx], n[idx])


def indices_nn(index, embs, idxs):
    assert embs.shape[0] == idxs.shape[0]

    if isinstance(idxs, torch.Tensor):
        idxs = idxs.detach().cpu().numpy()

    I = index.search(embs.detach(), 2)
    s = I != idxs.reshape(-1, 1)
    s[s.sum(1) > 1, 1] = False

    return I[s]


def nearest_k_embeddings(
    index, embs, labels, kind="all", return_centroids=True, k=1
):
    assert kind in ["all", "neg", "pos"], "unknown embedding kind"

    n = embs.shape[0]
    total = index.labels.shape[0]

    nn_labels = None
    nn_idxs = None
    nn_mask = torch.zeros((n, k)).bool()

    for m in range(2, total // k):
        nn_idxs = index.search(embs.detach().cpu().numpy(), k * m)
        nn_labels = index.labels.to(embs.device)[nn_idxs.flatten()].reshape(
            nn_idxs.shape
        )

        if kind == "all":
            nn_mask = torch.ones_like(nn_labels)
        elif kind == "neg":
            nn_mask = nn_labels != labels.unsqueeze(1)
        elif kind == "pos":
            nn_mask = nn_labels == labels.unsqueeze(1)

        # there is enough nearest neighbors
        if (nn_mask.sum(dim=1) >= k).all():
            break

    nn_idxs = torch.Tensor(nn_idxs).long()

    # reduce to k nearest neigbors
    for o in range(k, nn_mask.shape[1]):
        enough_neigbors = nn_mask[:, :o].sum(1) == k
        nn_mask[enough_neigbors, o:] = False

        if enough_neigbors.all():
            break

    nn_target_idxs = nn_idxs[nn_mask]

    # negatives tensor with the size:
    # <num_embeddings x k x dims_embedding>
    target_embeddings = (
        torch.Tensor(index.retrieve(nn_target_idxs.flatten()))
        .view(-1, k, embs.shape[1])
        .to(embs.device)
    )

    if return_centroids:
        return target_embeddings.mean(dim=1)

    return target_embeddings
