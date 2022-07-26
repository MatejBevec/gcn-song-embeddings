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

## THIS IS A TEMPORARY SETUP
## SHOULD MOVE ALL PREPARATION TO A SCRIPT AND JUST RUN
## prepare.py, pinsage_training.py, eval.py

DATA_DIR = "./dataset_final_intersect"
FEATURES_DIR = "./dataset_final_intersect/features_openl3"

def eval_baselines():
    # generate embeddings and knn list for all baselines
    # compute offline metrics
    # show results

    dataset = SpotifyGraph(DATA_DIR, None)
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    positives = dataset.load_positives(os.path.join(DATA_DIR, "positives_lfm.json"))
    train_pos, test_pos = dataset.load_positives_split(os.path.join(DATA_DIR, "positives_lfm.json"))
    n_items = torch.arange(0, len(track_ids))

    baselines = {
        "TrackTrackCfALS": TrackTrackCF(algo="als"),
        "ColTrackCfALS": ColTrackCF(algo="als"),
        "PageRank": PersPageRank(),
        "node2vec": FastNode2Vec(),
        "OpenL3": EmbLoader("dataset_final_intersect/features_openl3"),
        #"PinSageOpenL3LFMBestBest": EmbLoader("runs_gs4/gridsearch#0.0.0.0.0.1.0.0/emb"),
        ###"PinSagePageRank": EmbLoader("runs/pinsage_pagerank/emb"),
        "PinSageBase": PinSageWrapper(
            train_params={"T": 3, "lr": 0.0001, "epochs": 30, "n_layers": 2, "hard_negatives": False,
                            "decay": 0.95, "margin": 1e-05, "out_dim": 128},
            run_name="pinsage_base"
        ),


    }
    
    BL_DIR = "./baselines_final_intersect_again"
    knn_dict = eval.get_knn_dict(baselines, g, track_ids, train_pos, test_pos, features, BL_DIR)
    
    eval.crawl_embedding(knn_dict, track_ids, dataset, g,
                    model_names=["PinSageBase", "OpenL3", "TrackTrackCfALS", "ColTrackCfALS", "PageRank", "node2vec"])

    queries = [
        "2NBQ3xXu6js7xevPcVWTU2",
        "4UobpfzxcDNBplXZ4rQwpm",
        "1a31jGgyZ5c2d10CPWkGxM",
        "66LT15XEqCaWiMG44NGQRE",
        "40h65HAR8COEoqkMwUUQHu",
        "6m8vPz16wTojmAnVgdWtls",
        "2tJcd5q2oT3RmFtXJ0Ii6L"
    ]

    # eval.export_recommendation_lists(g, track_ids, dataset, queries, knn_dict,
    #             model_names=["PinSageOpenL3LFMBestBest", "OpenL3", "TrackTrackCfALS", "node2vec", "PageRank"])


if __name__ == "__main__":
        eval_baselines()
