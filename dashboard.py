import os
import sys
import torch

from dataset_creation.get_data import DatasetCollector
from spotify_graph import SpotifyGraph
from pinsage_training import PinSage
import pinsage_training as pt
import generate_node_features as gnf
import generate_positives as gp
from baselines import *
import eval

DATA_DIR = "./dataset_micro"
FEATURES_DIR = "./dataset_micro/features_openl3"

## THIS IS A TEMPORARY SETUP
## SHOULD MOVE ALL PREPARATION TO A SCRIPT AND JUST RUN
## prepare.py, pinsage_training.py, eval.py


def prepare_dataset():
    # (download clips)
    # download cover images
    # generate node features
    # generate positives

    DOWNLOAD_CLIPS = True
    NODE_FEAT = {
        "openl3": gnf.OpenL3()
    }

    dc = DatasetCollector(DATA_DIR, 0)
    dc.download_images()
    if DOWNLOAD_CLIPS:
        dc.download_clips()

    gnf.generate_features(DATA_DIR, NODE_FEAT, online = not DOWNLOAD_CLIPS)

    dataset = SpotifyGraph(DATA_DIR, FEATURES_DIR)
    
    gp.generate_positives_simple_walks(dataset, 5000, 3)
    

def train_pinsage():
    # precompute neighborhoods
    # train pinsage
    # save embeddings

    dataset = SpotifyGraph(DATA_DIR, FEATURES_DIR)
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    positives = dataset.load_positives(os.path.join(DATA_DIR, "positives.json"))
    n_items = torch.arange(0, len(track_ids))

    pinsage = PinSage(g, n_items, features, positives)
    pinsage.train() # parameters in pinsage_traning.py
    pt.save_embeddings(pinsage, dataset)

    pass

def eval_baselines():
    # generate embeddings and knn list for all baselines
    # compute offline metrics
    # show results

    dataset = SpotifyGraph(DATA_DIR, FEATURES_DIR)
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    positives = dataset.load_positives(os.path.join(DATA_DIR, "positives.json"))
    n_items = torch.arange(0, len(track_ids))

    baselines = {
        "Snore": Snore(),
        "node2vec": Node2Vec(),
        "PageRank": PersPageRank(),
        "PinSageHN": EmbLoader("runs/micro3/emb"),
        "PinSageL3": EmbLoader("runs/micro_openl3/emb"),
        "OpenL3": EmbLoader("dataset_micro/features_openl3"),
        "Preferential": Preferential(),
        "JaccardIndex": JaccardIndex()
    }
    
    knn_dict = eval.get_knn_dict(baselines, g, track_ids, positives, positives, features)
    results_table = eval.compute_results_table(knn_dict, positives)
    print(results_table)
    results_table.to_csv("./results.csv")


if __name__ == "__main__":

    action = sys.argv[1]
    
    if action == "prepare":
        prepare_dataset()

    if action == "train":
        train_pinsage()

    if action == "eval":
        eval_baselines()
