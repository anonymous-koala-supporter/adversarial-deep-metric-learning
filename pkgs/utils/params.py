import hashlib
import os
import tempfile

import numpy as np

import faiss
import torch


def get_ith_or_first(l, i, n: int = -1, type=None):
    l = str(l).split(",")

    if len(l) == 1:
        cand = l[0]
    else:
        cand = l[i]

    if cand == "none":
        return None

    if type is not None:
        return type(cand)

    return cand


def hash_from_params(params):
    d = dict(params)
    irrelevant_prefixes = ["pgd", "cw", "fgsm", "epsilon"]

    def is_irrelevant(s):
        for pre in irrelevant_prefixes:
            if s.startswith(pre):
                return True
        return False

    d = {k: d[k] for k in sorted(d.keys()) if not is_irrelevant(k)}

    hash_object = hashlib.sha256(str(d).encode())
    return hash_object.hexdigest()


class SearchIndex(object):
    def __init__(self, embedding_size):
        self.index = faiss.IndexFlatL2(embedding_size)
        self.labels = torch.Tensor([])

    def add(self, embeddings, labels):
        _np_embs = None

        if torch.is_tensor(embeddings):
            _np_embs = embeddings.cpu().numpy()
        elif isinstance(embeddings, np.ndarray):
            _np_embs = embeddings
        else:
            raise ValueError(f"unknown embedding type: {type(embeddings)}")

        self.index.add(_np_embs)
        self.labels = torch.cat([self.labels, labels])

    def search(self, embeddings, k):
        _np_embs = None
        if torch.is_tensor(embeddings):
            _np_embs = embeddings.cpu().numpy()
        elif isinstance(embeddings, np.ndarray):
            _np_embs = embeddings
        else:
            raise ValueError(f"unknown embedding type: {type(embeddings)}")

        _, idxs = self.index.search(_np_embs, k)

        return idxs

    def retrieve(self, idxs):
        if torch.is_tensor(idxs):
            idxs = idxs.long().numpy()

        embs = np.stack([self.index.reconstruct(int(i)) for i in idxs])

        return embs

    def _index_file(self, params):
        h = hash_from_params(params)
        return os.path.join(
            tempfile.gettempdir(), f"robust-emb.index.{h}.fidx"
        )

    def _labels_file(self, params):
        h = hash_from_params(params)
        return os.path.join(tempfile.gettempdir(), f"robust-emb.label.{h}.pt")

    def load(self, params):
        faiss_file = self._index_file(params)
        labels_file = self._labels_file(params)

        if not os.path.exists(faiss_file):
            return False

        if not os.path.exists(labels_file):
            return False

        self.index = faiss.read_index(faiss_file)
        self.labels = torch.load(labels_file)

        return True

    def save(self, params):
        faiss_file = self._index_file(params)
        labels_file = self._labels_file(params)

        if os.path.exists(faiss_file):
            return False

        if os.path.exists(labels_file):
            return False

        self.index = faiss.write_index(self.index, faiss_file)
        self.labels = torch.save(self.labels, labels_file)

        return True

    def __len__(self):
        return len(self.labels)

    def reset(self):
        self.index.reset()
        self.labels = torch.Tensor([])
