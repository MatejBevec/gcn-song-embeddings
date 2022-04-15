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

DATA_DIR = "./dataset_small"
FEATURES_DIR = "./dataset_small/features_openl3"

def prepare_dataset():
    # (download clips)
    # download cover images
    # generate node features
    # generate positives

    DOWNLOAD_CLIPS = True
    NODE_FEAT = {
        "openl3": gnf.OpenL3(),
        #"vggish": gnf.Vggish2(),
        #"musicnn": gnf.MusicNN()
    }

    dc = DatasetCollector(DATA_DIR, 0)
    dc.download_images()
    if DOWNLOAD_CLIPS:
        dc.download_clips()

    gnf.generate_features(DATA_DIR, NODE_FEAT, online = not DOWNLOAD_CLIPS)

    dataset = SpotifyGraph(DATA_DIR, FEATURES_DIR)
    
    gp.generate_positives(dataset)
    

def train_pinsage():
    # precompute neighborhoods
    # train pinsage
    # save embeddings

    dataset = SpotifyGraph(DATA_DIR, FEATURES_DIR)
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    positives = dataset.load_positives(os.path.join(DATA_DIR, "positives.json"))
    #n_items = torch.arange(0, len(track_ids))

    pinsage = PinSage(g, len(track_ids), features, positives)
    pinsage.train() # parameters in pinsage_traning.py
    pt.save_embeddings(pinsage, dataset)


def eval_baselines():
    # generate embeddings and knn list for all baselines
    # compute offline metrics
    # show results

    dataset = SpotifyGraph(DATA_DIR, FEATURES_DIR)
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    positives = dataset.load_positives(os.path.join(DATA_DIR, "positives_lfm_large.json"))
    train_pos, test_pos = dataset.load_positives_split(os.path.join(DATA_DIR, "positives_lfm_large.json"))
    n_items = torch.arange(0, len(track_ids))

    baselines = {
        "Random": Random(),
        "RandomFeatures": EmbLoader("dataset_small/features_random"),
        #"Snore": Snore(),
        "node2vec": FastNode2Vec(),
        "PageRank": PersPageRank(),
        "OpenL3": EmbLoader("dataset_small/features_openl3"),
        "VGGish": EmbLoader("dataset_small/features_vggish_msd"),
        "MusicNN": EmbLoader("dataset_small/features_musicnn"),
        # "Preferential": Preferential(),
        # "JaccardIndex": JaccardIndex(),
        "ColTrackCfALS": ColTrackCF(algo="als"),
        "TrackTrackCfALS": TrackTrackCF(algo="als"),
        ##"ColTrackCfBPR": ColTrackCF(algo="bpr"),
        #"TrackTrackCfBPR": TrackTrackCF(algo="bpr"),
        ##"ColTrackCfLMF": ColTrackCF(algo="lmf"),
        #"TrackTrackCfLMF": TrackTrackCF(algo="lmf"),
        #"UnsupervisedGraphSAGE": GraphSAGE()

        "PinSageOpenL3": EmbLoader("runs/small_openl3_pr3/emb"),
        #"PinSageOpenL3t10": EmbLoader("runs/small_openl3_T10/emb"),
        #"PinSageOpenL3LFM": EmbLoader("runs/small_openl3_lfm_test/emb"),
        "PinSageOpenL3LFMfull": EmbLoader("runs/small_openl3_lfm/emb"),
        #"PinSageOpenL3LFMlarge": EmbLoader("runs/small_openl3_lfm_large/emb"),
        # "PinSageOpenl3Mixed": EmbLoader("runs/small_openl3_mixed/emb"),
        ##"PinSageVggishLFMlarge": EmbLoader("runs/small_vggish_lfm_large/emb"),
        ##"PinSageMusicNNLFMlarge": EmbLoader("runs/small_musicnn_lfm_large_2/emb"),
        "PinSageOpenL3longLFMlarge": EmbLoader("runs/small_openl3_lfm_large_2/emb"),
        #"PinSageRandomLFMlarge": EmbLoader("runs/small_random_lfm_large/emb"),
        "PinSageT10OpenL3LFMlarge": EmbLoader("runs/small_openl3_lfm_large_3/emb"),
        # "PinSageVggish": EmbLoader("runs/small_vggish/emb"),
        # "PinSageMusicNN": EmbLoader("runs/small_musicnn/emb"),
        "PinSageOpenL3LFMBest": EmbLoader("runs_gs1/gridsearch#0.1.1.0.1/emb")
    }
    
    BL_DIR = "./baselines_lfm"
    knn_dict = eval.get_knn_dict(baselines, g, track_ids, train_pos, test_pos, features, BL_DIR)
    results_table = eval.compute_results_table(knn_dict, test_pos, g)
    print(results_table)
    results_table.to_csv("./results_lfm.csv")

    # beyond_table = eval.compute_beyond_accuracy_table(knn_dict, test_pos, g, features)
    # print(beyond_table)
    # beyond_table.to_csv("./results_beyond_lfm.csv")

    #eval.crawl_embedding(knn_dict, track_ids, dataset, g, ["PinSageOpenL3", "PinSageOpenL3LFMfull"])


if __name__ == "__main__":

    action = sys.argv[1]
    
    if action == "prepare":
        prepare_dataset()

    if action == "train":
        train_pinsage()

    if action == "eval":
        eval_baselines()
