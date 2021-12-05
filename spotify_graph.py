import os
from os import path
import json

import pandas as pd
import numpy as np
import dgl
import torch

class SpotifyGraph():

    def __init__(self, dir, features_dir):

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
        
        print("Loading node features...")
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
        
        # vectors of "to" and "from" nodes for all edges
        # bidirectional duplicates are included in self.graph["edges"]
        index_map = {nid: i for i, nid in enumerate(all_ids)}

        from_nodes = [ index_map[e["from"]] for e in self.graph["edges"] ]
        to_nodes = [ index_map[e["to"]] for e in self.graph["edges"]]

        g.add_edges(from_nodes, to_nodes)

        if self.ft_dir:
            for fn in os.listdir(self.ft_dir):
                node_id = fn.rsplit('.')[0]
                self.features_dict[node_id] = torch.load(os.path.join(self.ft_dir, fn))
            features = torch.stack(list(self.features_dict.values()), dim=0)
        else:
            features = None

        #self.g, self.track_ids, self.col_ids, self.features = g, track_ids, col_ids, features
        return g, track_ids, col_ids, features

    def load_positives(self, pos_pth):

        with open(pos_pth, "r", encoding="utf-8") as f:
            positives = json.load(f)

        index_map = {nid: i for i, nid in enumerate(list(self.tracks))}

        a = torch.tensor([ index_map[pair["a"]] for pair in positives ], dtype=torch.int64)
        b = torch.tensor([ index_map[pair["b"]] for pair in positives ], dtype=torch.int64)
        return torch.stack((a,b), dim=1)

    def load_batch_features(self, ids):
        
        batch_features = {}
        for node_id in ids:
            batch_features[node_id] = torch.load(
                os.path.join(self.ft_dir, node_id, ".pt"))

        return batch_features


if __name__ == "__main__":

    dataset = SpotifyGraph("./dataset_mini", "./dataset_mini/features_mfcc")
    g, track_ids, col_ids, _ = dataset.to_dgl_graph()

    positives = dataset.load_positives("./dataset_mini/positives.json")
