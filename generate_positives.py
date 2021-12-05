import pandas as pd
import numpy as np
import dgl
import torch
import json
from pinsage_code.pinsage_training import PRECOMP_NAME

from spotify_graph import SpotifyGraph
import pinsage_model as psm

PRECOMP_NAME = "./neighborhoods_micro.pt"

def generate_positives_simple_walks(dataset, m, T):

    g, track_ids, col_ids, _ = dataset.to_dgl_graph()
    _, nbhds = psm.precompute_neighborhoods_topt(g,
        len(track_ids),
        psm.DEF_HOPS,
        psm.DEF_ALPHA,
        psm.DEF_T_PRECOMP,
        PRECOMP_NAME)

    rnd_ids = torch.randint(0, len(track_ids), (m,))
    rnd_rank = torch.randint(0, T, (m,))

    print(rnd_ids, rnd_rank)
    
    positives = []

    # screw it, a for loop will do
    for i,id in enumerate(rnd_ids):
        a = track_ids[id]
        b_i = nbhds[id,rnd_rank[i]]
        b = track_ids[b_i]
        positives.append( {"a": a, "b": b} )

        print(f"{dataset.tracks[a]['name']} - {dataset.tracks[a]['artist']}")
        print(f"{dataset.tracks[b]['name']} - {dataset.tracks[b]['artist']}")
        print()

    #output = list(set(positives))[0:n]
    return positives

if __name__ == "__main__":
    
    dataset = SpotifyGraph("./dataset_micro", None)
    positives = generate_positives_simple_walks(dataset, 5000, 3)

    save_path = "./dataset_micro/positives.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(positives, f, ensure_ascii=False, indent=2) 
