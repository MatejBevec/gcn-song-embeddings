import os
from os import path
import time
import math
from abc import ABC, abstractmethod

import numpy as np
import torch

from baselines import cosine_sim_ab


def _gen_pl_random_seeds(qids, qemb, knn, m,  k, weighted):
    assert len(qids) * k > m

    # slow version - could loop
    gids = set()
    while len(gids) < m:
        seed_track = np.random.randint(0, len(qids), 1).item()
        i = np.asscalar(np.random.randint(0, k, 1))
        # Select a seed track and take its i-th nearest neighbor
        print(seed_track)
        print(i)
        print(qids[seed_track])
        added_id = knn[qids[seed_track], i].item()
        gids.add(added_id)
        
    return gids

def _gen_pl_mean_seed(qids, qemb, knn, m, k,  weighted):
    assert k > m

    mean_vec = torch.mean(qemb, dim=0)
    mean_vec_repeated = mean_vec.repeat(len(qids), 1)
    print(mean_vec_repeated)
    print(qemb)
    cos_sim = cosine_sim_ab(mean_vec_repeated, qemb)[0, :]
    print(cos_sim)
    seed_track = torch.argmin(cos_sim, dim=0).item()
    print(seed_track)

    offsets = torch.randint(0, k, (m,))
    gids_tensor = knn[qids[seed_track], offsets]
    print(gids_tensor)
    return set(gids_tensor.tolist())

def _gen_pl_cluster_seeds(qids, qemb, knn, m, k, weighted):
    pass

SEED_FUNCS = {
    "random": _gen_pl_random_seeds,
    "mean": _gen_pl_mean_seed,
    "cluster": _gen_pl_cluster_seeds
}

def generate_playlist_from_emb(query_ids, query_emb, knn,
                    seed_style="random", k=3, weighted=False):    
    func = SEED_FUNCS[seed_style]
    generated_ids = func(query_ids, query_emb, knn, k, weighted)
    return torch.tensor(list(generated_ids))



class PlaylistModel(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, g, track_ids, features, train_positivies, train_playlists):
        pass

    @abstractmethod
    def generate_playlist(self, query_playlist):
        pass


if __name__ == "__main__":

    qids = torch.tensor([0,1,2,3])

    qemb = torch.tensor([
        [1, 1, 0, 0],
        [0, 0, 0, 2],
        [3, -2, -3, -4],
        [2, 1, 2, 1]
    ]).float()
    
    knn = torch.tensor([
        [9, 19, 34, 32, 78, 95, 133, 3, 4, 5],
        [95, 97, 19, 9, 55, 66, 5, 1, 2, 100],
        [5, 4, 424, 23, 95, 22, 65, 543, 20, 21],
        [0, 133, 7, 12, 13, 14, 15, 16, 17, 18]
    ])

    gids = _gen_pl_mean_seed(qids, qemb, knn, 5, 10, False)
    print(gids)
