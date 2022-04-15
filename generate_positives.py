import os
import pandas as pd
import numpy as np
import dgl
import torch
import json

from spotify_graph import SpotifyGraph
import pinsage_model as psm

#PRECOMP_NAME = "./neighborhoods_small.pt"

def generate_positives_simple_walks(dataset, m, T):

    print(f"\033[0;33mGenerating positive training pairs for {dataset.base_dir}\
         with Personalized PageRank...\033[0m")

    g, track_ids, col_ids, _ = dataset.to_dgl_graph()
    _, nbhds = psm.precompute_neighborhoods_topt(g,
        len(track_ids),
        psm.DEF_HOPS,
        psm.DEF_ALPHA,
        psm.DEF_T_PRECOMP,
        dataset.nbhds_path)

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

        #print(f"{dataset.tracks[a]['name']} - {dataset.tracks[a]['artist']}")
        #print(f"{dataset.tracks[b]['name']} - {dataset.tracks[b]['artist']}")
        #print()

    #output = list(set(positives))[0:n]
    return positives

def generate_positives(dataset_dir, n="auto", T=3):

    dataset = SpotifyGraph(dataset_dir, None)
    n = len(dataset.tracks)*2 if n == "auto" else n
    print(n)
    positives = generate_positives_simple_walks(dataset, n, T)

    save_path = os.path.join(dataset_dir, "positives.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(positives, f, ensure_ascii=False, indent=2)    

def generate_random_positives(dataset_dir, n="auto"):

    dataset = SpotifyGraph(dataset_dir, None)
    tracks = list(dataset.tracks.keys())
    n = len(tracks)*2 if n == "auto" else n
    rand_a = torch.randint(0, len(tracks), (n,))
    rand_b = torch.randint(0, len(tracks), (n,))
    positives = []

    for i in range(n):
        positives.append({
            "a": tracks[rand_a[i]],
            "b": tracks[rand_b[i]]
        })

    save_path = os.path.join(dataset_dir, "positives_random.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(positives, f, ensure_ascii=False, indent=2)    

if __name__ == "__main__":
    
    generate_random_positives("dataset_small")
