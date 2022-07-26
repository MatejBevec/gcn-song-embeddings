import os
import sys
import torch

from dataset_creation.get_data import DatasetCollector
from spotify_graph import SpotifyGraph
from pinsage_training import PinSage
import pinsage_training as pt
import generate_node_features as gnf
import generate_positives as gp
import generate_positives_lfm as gplfm
from baselines import *
import eval



DATA_DIR = "./dataset_final_intersect"
FEATURES_DIR = "./dataset_final_intersect/features_openl3"

def prepare_dataset():
    """Prepare all necessary training data."""

    # (download clips)
    # download cover images
    # generate node features
    # generate positives

    #DOWNLOAD_CLIPS = True
    # NODE_FEAT = {
    #     "openl3": gnf.OpenL3(),
    #     "vggish": gnf.Vggish2(model="MSD_vgg"),
    #     "musicnn": gnf.MusicNN()
    # }
    #if DOWNLOAD_CLIPS:
    #    dc.download_clips()
    #gnf.generate_features(DATA_DIR, NODE_FEAT, online = not DOWNLOAD_CLIPS)

    dc = DatasetCollector(DATA_DIR, 0)
    dc.download_images()

    dataset_openl3 = SpotifyGraph(DATA_DIR, None)
    dataset_vggish = SpotifyGraph(DATA_DIR, "./dataset_final_intersect/features_vggish")
    dataset_musicnn = SpotifyGraph(DATA_DIR, "./dataset_final_intersect/features_musicnn")
    
    #gp.generate_positives(dataset, 200000)
    #gplfm.generate_lfm_positives(dataset, 5000000)

    

def train_pinsage():
    """Train PinSage models 'manually'."""

    # precompute neighborhoods
    # train pinsage
    # save embeddings

    
    dataset_vggish = SpotifyGraph(DATA_DIR, "./dataset_final_intersect/features_vggish")
    dataset_musicnn = SpotifyGraph(DATA_DIR, "./dataset_final_intersect/features_musicnn")

    # Train on 3 models on 3 different node features
    features = {
        "vggish": "./dataset_final_intersect/features_vggish",
        "musicnn": "./dataset_final_intersect/features_musicnn",
        "random": "./dataset_final_intersect/features_random",
    }

    for ft in features:
        dataset = SpotifyGraph(DATA_DIR, features[ft])
        g, track_ids, col_ids, features = dataset.to_dgl_graph()
        positives = dataset.load_positives(os.path.join(DATA_DIR, "positives_lfm.json"))
        pinsage = PinSage(g, len(track_ids), features, positives)
        setattr(pinsage, "run_name", f"pinsage_{ft}_ft")
        pinsage.train() # SEE HYPERPARAMETERS IN pinsage_training.py
        pt.save_embeddings(pinsage, dataset)

    # Train model on alternative (PPR) positives
    dataset = SpotifyGraph(DATA_DIR, FEATURES_DIR)
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    positives = dataset.load_positives(os.path.join(DATA_DIR, "positives.json")) #positives.json = PPR positives
    pinsage = PinSage(g, len(track_ids), features, positives)
    setattr(pinsage, "run_name", f"pinsage_pagerank_pos")
    pinsage.train() # SEE HYPERPARAMETERS IN pinsage_training.py
    pt.save_embeddings(pinsage, dataset)


def eval_baselines():
    """Train and evaluate baseline methods on a related song prediction task."""

    # generate embeddings and knn list for all baselines
    # compute offline metrics
    # show results

    dataset = SpotifyGraph(DATA_DIR, FEATURES_DIR)
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    positives = dataset.load_positives(os.path.join(DATA_DIR, "positives_lfm.json"))
    train_pos, test_pos = dataset.load_positives_split(os.path.join(DATA_DIR, "positives_lfm.json"))
    n_items = torch.arange(0, len(track_ids))

    # baselines to train and eval on 'g', 'feature' and 'positives'
    baselines = {
        
        "Random": Random(),

        "ColTrackCfALS": ColTrackCF(algo="als"),
        "TrackTrackCfALS": TrackTrackCF(algo="als"),
        "PageRank": PersPageRank(),
        "node2vec": FastNode2Vec(),

        "OpenL3": EmbLoader("dataset_final_intersect/features_openl3"),
        "VGGish": EmbLoader("dataset_final_intersect/features_vggish"),
        "MusicNN": EmbLoader("dataset_final_intersect/features_musicnn"),
        
        "PinSageBase": PinSageWrapper(
            train_params={"T": 3, "lr": 0.0001, "epochs": 30, "n_layers": 2, "hard_negatives": False,
                            "decay": 0.95, "margin": 1e-05, "out_dim": 128},
            run_name="pinsage_base"
        ),

        "PinsageRandom": EmbLoader("runs/pinsage_random_ft/emb"),
        "PinsagePageRank": EmbLoader("runs/pinsage_pagerank_pos/emb"),
        "PinsageMusicNN": EmbLoader("runs/pinsage_musicnn_ft/emb"),
        "PinsageVggish": EmbLoader("runs/pinsage_vggish_ft/emb"),

        "PinSageHardNeg": PinSageWrapper(
            train_params={"T": 3, "lr": 0.0001, "epochs": 30, "n_layers": 2, "hard_negatives": True,
                            "hn_min": 10, "hn_max":100, "decay": 0.95, "margin": 1e-05, "out_dim": 128},
            run_name="pinsage_hard_neg",
            log=False
        ),

        "PinSageFewEpochs": PinSageWrapper(
            train_params={"T": 3, "lr": 0.0001, "epochs": 5, "n_layers": 2, "hard_negatives": False,
                            "decay": 0.95, "margin": 1e-05, "out_dim": 128},
            run_name="pinsage_few_epochs"
        ),

        "PinSageManyParams": PinSageWrapper(
            train_params={"T": 3, "lr": 0.0001, "epochs": 30, "n_layers": 4, "hard_negatives": False,
                            "decay": 0.95, "margin": 1e-05, "out_dim": 256, "hidden_dim": 1024},
            run_name="pinsage_many_params",
            log=False
        ),  

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

    }

    
    BL_DIR = "./baselines_final_intersect_again"
    knn_dict = eval.get_knn_dict(baselines, g, track_ids, train_pos, test_pos, features, BL_DIR)

    results_table = eval.compute_results_table(knn_dict, test_pos, g, times=False)
    results_table.to_csv("./results_final_intersect_again.csv")

    beyond_table = eval.compute_beyond_accuracy_table(knn_dict, test_pos, g, features)
    beyond_table.to_csv("./results_final_intersect_again_beyond.csv")

    print(results_table)
    print(beyond_table)
    
    # Explore recommendation examples
    eval.crawl_embedding(knn_dict, track_ids, dataset, g,
                    ["PinSageBase", "OpenL3", "TrackTrackCfALS", "node2vec", "PageRank"])


if __name__ == "__main__":

    action = sys.argv[1]
    
    if action == "prepare":
        prepare_dataset()

    if action == "train":
        train_pinsage()

    if action == "eval":
        eval_baselines()

    if action == "all":
        prepare_dataset()
        train_pinsage()
        eval_baselines()

