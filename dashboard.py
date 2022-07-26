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
    positives = dataset.load_positives(os.path.join(DATA_DIR, "positives_lfm.json"))
    train_pos, test_pos = dataset.load_positives_split(os.path.join(DATA_DIR, "positives_lfm.json"))
    n_items = torch.arange(0, len(track_ids))

    baselines = {
        "Random": Random(),
        ####"ColTrackCfALS": ColTrackCF(algo="als"),
        ####"TrackTrackCfALS": TrackTrackCF(algo="als"),
        ####"PageRank": PersPageRank(),
        ####"node2vec": FastNode2Vec(),

        #"OpenL3": EmbLoader("dataset_final_intersect/features_openl3"),
        #"VGGish": EmbLoader("dataset_small/features_vggish_msd"),
        #"MusicNN": EmbLoader("dataset_small/features_musicnn"),


        #"RandomFeatures": EmbLoader("dataset_small/features_random"),
        #"ColTrackCfBPR": ColTrackCF(algo="bpr"),
        #"TrackTrackCfBPR": TrackTrackCF(algo=W"bpr"),
        #"ColTrackCfLMF": ColTrackCF(algo="lmf"),
        #"TrackTrackCfLMF": TrackTrackCF(algo="lmf"),

        #"Snore": Snore(),
        #"JaccardFast": JaccardFast(),
        #"Preferential": Preferential(),
        #"JaccardIndex": JaccardIndex(),
        #"AdamicAdar": AdamicAdar(),
        #"UnsupervisedGraphSAGE": GraphSAGE()
        

        # THE MODELS BELOW NEED ANOTHER TRAINING ON TRAIN/TEST SPLIT

        ###"PinSageRandomLFMlarge": EmbLoader("runs/small_random_lfm_large/emb"),  
        ###"PinSageOpenL3LFM": EmbLoader("runs/small_openl3_lfm_test/emb"), #lfm_mini
        ###"PinSageOpenL3": EmbLoader("runs/small_openl3_pr3/emb"),
        #"PinSageOpenL3t10": EmbLoader("runs/small_openl3_T10/emb"),
        #"PinSageOpenL3LFMfull": EmbLoader("runs/small_openl3_lfm/emb"),
        #"PinSageOpenL3LFMlarge": EmbLoader("runs/small_openl3_lfm_large/emb"),
        # "PinSageOpenl3Mixed": EmbLoader("runs/small_openl3_mixed/emb"),
        ##"PinSageVggishLFMlarge": EmbLoader("runs/small_vggish_lfm_large/emb"),
        ##"PinSageMusicNNLFMlarge": EmbLoader("runs/small_musicnn_lfm_large_2/emb"),
        #"PinSageOpenL3longLFMlarge": EmbLoader("runs/small_openl3_lfm_large_2/emb"),
        ##"PinSageT10OpenL3LFMlarge": EmbLoader("runs/small_openl3_lfm_large_3/emb"),
        # "PinSageVggish": EmbLoader("runs/small_vggish/emb"),
        # "PinSageMusicNN": EmbLoader("runs/small_musicnn/emb"),
        ##"PinSageOpenL3LFMBest": EmbLoader("runs_gs1/gridsearch#0.1.1.0.1/emb"),
        ####"PinSageOpenL3LFMBestBest": EmbLoader("runs_gs4/gridsearch#0.0.0.0.0.1.0.0/emb"),
        ####"PinSagePageRank": EmbLoader("runs/pinsage_pagerank/emb"),
        #"PinSageRandomFeatures": EmbLoader("runs/pinsage_random/emb")

        # RETRAINED MODELS ABOVE

        ####"PinSageBase2": EmbLoader("runs/pinsage_best_intersect/emb"),

        #"PinsageRandom": EmbLoader("runs/pinsage_randomft_intersect/emb"),
        # "PinsagePageRank": EmbLoader("runs/pinsage_pagerank_intersect/emb"),
        # "PinsageMusicNN": EmbLoader("runs/pinsage_musicnn_intersect/emb"),
        # "PinsageVggish": EmbLoader("runs/pinsage_vggish_intersect/emb"),
        
        #"OpenL3": EmbLoader("dataset_final_intersect/features_openl3"),
        #"VGGish": EmbLoader("dataset_final_intersect/features_vggish"),
        #"MusicNN": EmbLoader("dataset_final_intersect/features_musicnn"),

        # TEST
        # "PinSageTest": PinSageWrapper(
        #     train_params={"T": 3, "lr": 0.0001, "epochs": 3, "n_layers": 2, "hard_negatives": False,
        #                     "decay": 0.95, "margin": 1e-05, "out_dim": 128},
        #     run_name="pinsage_test"
        # ),

        #ABLATION STUDY
        # "PinSageBase": PinSageWrapper(
        #     train_params={"T": 3, "lr": 0.0001, "epochs": 30, "n_layers": 2, "hard_negatives": False,
        #                     "decay": 0.95, "margin": 1e-05, "out_dim": 128},
        #     run_name="pinsage_base"
        # ),

        "PinSageHardNeg": PinSageWrapper(
            train_params={"T": 3, "lr": 0.0001, "epochs": 30, "n_layers": 2, "hard_negatives": True,
                            "hn_min": 10, "hn_max":100, "decay": 0.95, "margin": 1e-05, "out_dim": 128},
            run_name="pinsage_hard_neg",
            log=False
        ),

        # # "PinSageFewEpochs": PinSageWrapper(
        # #     train_params={"T": 3, "lr": 0.0001, "epochs": 5, "n_layers": 2, "hard_negatives": False,
        # #                     "decay": 0.95, "margin": 1e-05, "out_dim": 128},
        # #     run_name="pinsage_few_epochs"
        # # ),

        # # "JaccardIndex": JaccardIndex(projected=False),


        # "PinSageManyParams": PinSageWrapper(
        #     train_params={"T": 3, "lr": 0.0001, "epochs": 30, "n_layers": 4, "hard_negatives": False,
        #                     "decay": 0.95, "margin": 1e-05, "out_dim": 256, "hidden_dim": 1024},
        #     run_name="pinsage_many_params",
        #     log=False
        # ),  

    }
    
    BL_DIR = "./baselines_final_intersect_again"
    knn_dict = eval.get_knn_dict(baselines, g, track_ids, train_pos, test_pos, features, BL_DIR)

    results_table = eval.compute_results_table(knn_dict, test_pos, g, times=False)
    results_table.to_csv("./results_final_intersect_again.csv")

    beyond_table = eval.compute_beyond_accuracy_table(knn_dict, test_pos, g, features)
    beyond_table.to_csv("./results_final_intersect_again_beyond.csv")

    print(results_table)
    print(beyond_table)
    
    # eval.crawl_embedding(knn_dict, track_ids, dataset, g,
    #                 ["PinSageBase", "OpenL3", "TrackTrackCfALS", "node2vec", "PageRank"])


if __name__ == "__main__":

    action = sys.argv[1]
    
    if action == "prepare":
        prepare_dataset()

    if action == "train":
        train_pinsage()

    if action == "eval":
        eval_baselines()
