import os
from os import path
import json

import pandas as pd
import numpy as np
import dgl
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix, lil_matrix
import matplotlib.pyplot as plt



class SpotifyGraph():

    def __init__(self, dir, features_dir):
        
        self.base_dir = dir
        self.nbhds_path = os.path.join(self.base_dir, "neighborhoods.pt")
        self.tracks_pth = path.join(dir, "tracks.json")
        self.col_pth = path.join(dir, "collections.json")
        self.graph_pth = path.join(dir, "graph.json")

        self.img_dir = path.join(dir, "images")
        self.clip_dir = path.join(dir, "clips")

        print("Loading graph...")
        with open(self.tracks_pth, "r", encoding="utf-8") as f:
            self.tracks = json.load(f)
        with open(self.col_pth, "r", encoding="utf-8") as f:
            self.collections = json.load(f)
        with open(self.graph_pth, "r", encoding="utf-8") as f:
            self.graph = json.load(f)
        
        # todo: check if all nodes have features
        # todo: load batches into memory instead of whole dataset
        self.ft_dir = features_dir
        self.features_dict = {}

    def to_dgl_graph(self):

        track_ids = list(self.tracks)
        col_ids = list(self.collections)
        all_ids = track_ids.copy()
        all_ids.extend(col_ids)

        g = dgl.DGLGraph()
        n_nodes = len(track_ids) + len(col_ids)
        g.add_nodes(n_nodes)

        # temporary bodge
        g.base_dir = self.base_dir
        g.nbhds_path = self.nbhds_path

        # vectors of "to" and "from" nodes for all edges
        # bidirectional duplicates are included in self.graph["edges"]
        index_map = {nid: i for i, nid in enumerate(all_ids)}

        from_nodes = [ index_map[e["from"]] for e in self.graph["edges"] ]
        to_nodes = [ index_map[e["to"]] for e in self.graph["edges"]]

        g.add_edges(from_nodes, to_nodes)
        # BUG: why is this a DGLHeterorgraph??

        if self.ft_dir:
            pbar = tqdm(total=len(list(track_ids)), desc="Loading node features")
            features_list = []
            for i, track_id in enumerate(track_ids):
                pbar.update(1000) if i % 1000 == 0 else None
                #print(f"{i} done") if i % 1000 == 0 else None
                vec = torch.load(os.path.join(self.ft_dir, track_id + ".pt"))
                features_list.append(vec)
            features = torch.stack(features_list, dim=0)
            pbar.close()

            mean = features.mean(dim=0)
            std = features.std(dim=0, unbiased=True) + 1e-12
            features = (features - mean) / std

        else:
            features = None

        self.g, self.track_ids, self.col_ids, self.features = g, track_ids, col_ids, features
        return g, track_ids, col_ids, features


    def load_positives(self, pos_pth):

        with open(pos_pth, "r", encoding="utf-8") as f:
            positives = json.load(f)

        index_map = {nid: i for i, nid in enumerate(list(self.tracks))}

        a = torch.tensor([ index_map[pair["a"]] for pair in positives ], dtype=torch.int64)
        b = torch.tensor([ index_map[pair["b"]] for pair in positives ], dtype=torch.int64)
        perm = torch.randperm(a.shape[0])
        pos = torch.stack((a,b), dim=1)
        self.positives = pos
        return pos

    def load_positives_split(self, pos_pth, split=0.7, shuffle=True, random_seed=42):
        pos = self.load_positives(pos_pth)
        n = pos.shape[0]
        if shuffle:
            index = np.random.RandomState(random_seed).permutation(n)
            pos = pos[index, :]
        cut_point = int(split*n)
        train, test = pos[:cut_point, :], pos[cut_point:, :]
        return train, test

    def load_batch_features(self, ids):
        
        batch_features = {}
        for node_id in ids:
            batch_features[node_id] = torch.load(
                os.path.join(self.ft_dir, node_id, ".pt"))

        return batch_features

    def song_info(self, index_id):
        track_ids = list(self.tracks)
        name = self.tracks[track_ids[index_id]]["name"]
        artist = self.tracks[track_ids[index_id]]["artist"]
        return f"{name} - {artist}"





def _to_track_track_matrix(ids, positives):
    n = len(ids)
    pos_tuples = [(positives[i, 0], positives[i, 1]) for i in range(positives.shape[0])]
    pos_tuples.sort(key=lambda x: x[1])
    mat = lil_matrix((n, n), dtype=np.int32)
    for (a, b) in pos_tuples:
        if mat[a, b]:
            mat[a, b] += 1
        else:
            mat[a, b] = 1
    mat = mat.tocsr(copy=True)
    return mat

def get_positives_deg_dist(track_ids, g, positives, repeats=True):
    if repeats:
        degrees = g.in_degrees(torch.flatten(positives))
    else:
        ids_in_positives = torch.unique(positives)
        degrees = g.in_degrees(ids_in_positives)

    print("a")
    return degrees.numpy(), np.unique(degrees.numpy(), return_counts=True)

def get_graph_deg_dist(track_ids, g, positives):
    degrees = g.in_degrees(track_ids)
    return degrees.numpy(), np.unique(degrees.numpy(), return_counts=True)

def get_positives_cooccurence_dist(track_ids, g, positives):
    # with repeats!
    graph_co = get_graph_cooccurence_dist(track_ids, g, positives)[0]
    positives_co = graph_co[positives.flatten().numpy()]
    return positives_co, np.unique(positives_co, return_counts=True)

def get_graph_cooccurence_dist(track_ids, g, positives):
    mat = _to_track_track_matrix(track_ids, positives)
    co =  np.asarray(np.sum(mat, axis=1))
    return co, np.unique(co, return_counts=True)

def get_graph_positives_intersect(track_ids, g, positives):
    ids_in_positives = torch.unique(positives)
    intersect = set(ids_in_positives.tolist()) & set(track_ids)
    return np.array(list(intersect))

def print_dataset_stats(g, track_ids, positives):

    print("Basic dataset stats:")
    ids = np.arange(0, len(track_ids))

    print("\nNodes in graph: ", len(g))
    print("Songs in graph: ", len(ids))
    print("Playlists in graph: ", len(g) - len(ids))
    song_degrees = g.in_degrees(ids)
    print(type(song_degrees))
    print("Mean song degree: ", torch.mean(song_degrees.float()).item())
    print("Median song degree: ", torch.median(song_degrees).item())

    print("\n Positives: ", positives.shape[0])
    print("Unique songs in positives: ", len(torch.unique(positives)))
    co_counts, _ = get_graph_cooccurence_dist(ids, g, positives)
    print("Mean co-occurence count: ", np.mean(co_counts))
    print("Median co-occurence count: ", np.median(co_counts))

    intersect = get_graph_positives_intersect(ids, g, positives)
    print("Unique songs present in graph AND positives: ", len(intersect))


def save_dataset_distributions(g, track_ids, positives):

    ids = torch.arange(0, len(track_ids)).numpy()
    raw, (levels, counts) = get_positives_deg_dist(ids, g, positives, repeats=True)
    df = pd.DataFrame((levels, counts))
    df.to_csv("pos_deg_repeats.csv")
    raw, (levels, counts) = get_positives_deg_dist(ids, g, positives, repeats=False)
    df = pd.DataFrame((levels, counts))
    df.to_csv("pos_deg.csv")
    raw, (levels, counts) = get_graph_deg_dist(ids, g, positives)
    df = pd.DataFrame((levels, counts))
    df.to_csv("graph_deg.csv")
    raw, (levels, counts) = get_positives_cooccurence_dist(ids, g, positives)
    df = pd.DataFrame((levels, counts))
    df.to_csv("pos_co.csv")
    raw, (levels, counts) = get_graph_cooccurence_dist(ids, g, positives)
    df = pd.DataFrame((levels, counts))
    df.to_csv("graph_co.csv")




if __name__ == "__main__":

    dataset = SpotifyGraph("./dataset_final_intersect", "./dataset_final_intersect/features_openl3")
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    positives = dataset.load_positives("./dataset_final_intersect/positives_lfm.json")

    train, test = dataset.load_positives_split("./dataset_small/positives_lfm.json")

    print_dataset_stats(g, track_ids, positives)
    save_dataset_distributions(g, track_ids, positives)



