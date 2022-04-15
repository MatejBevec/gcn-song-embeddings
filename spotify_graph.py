import os
from os import path
import json

import pandas as pd
import numpy as np
import dgl
import torch
from tqdm import tqdm

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

    def to_col_track_matrix(self):
        if not self.g:
            self.to_dgl_graph()

        n_tracks, n_cols = len(track_ids), len(col_ids)
        mat = np.zeros((n_cols, n_tracks))
        for col in range(n_tracks, n_tracks+n_cols):
            neighbors = list( set(g.successors(col)) | set(g.predecessors(col)) )
            mat[col-n_tracks, neighbors] = 1
        return mat

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

    def to_track_track_matrix(self, track_ids, positives):   
        n = len(track_ids)
        mat = np.zeros((n, n))
        for i in range(positives.shape[0]):
            a = positives[i, 0]
            b = positives[i, 1]
            mat[a, b] = 1
        return mat


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


if __name__ == "__main__":

    dataset = SpotifyGraph("./dataset_micro", "./dataset_micro/features_openl3")
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    positives = dataset.load_positives("./dataset_micro/positives.json")

    train, test = dataset.load_positives_split("./dataset_micro/positives.json")
    print(train.shape)
    print(test.shape)