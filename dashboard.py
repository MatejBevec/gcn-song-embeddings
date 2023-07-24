import os
import sys

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

def prepare_dataset():
    """Prepare all necessary training data."""

    # (download clips)
    # download cover images
    # generate node features
    # generate positives

    dc = DatasetCollector(DATA_DIR, 0)
    dc.download_images()

    DOWNLOAD_CLIPS = True
    NODE_FEAT = {
        "openl3": gnf.OpenL3(),
        "vggish": gnf.Vggish2(model="MSD_vgg"),
        "musicnn": gnf.MusicNN()
    }
    if DOWNLOAD_CLIPS:
       dc.download_clips()
    gnf.generate_features(DATA_DIR, NODE_FEAT, online = not DOWNLOAD_CLIPS)

    dataset = SpotifyGraph(DATA_DIR, None)
    #dataset_vggish = SpotifyGraph(DATA_DIR, "./dataset_final_intersect/features_vggish")
    #dataset_musicnn = SpotifyGraph(DATA_DIR, "./dataset_final_intersect/features_musicnn")
    
    gplfm.generate_lfm_positives(dataset, 5000000)
    #gp.generate_positives(dataset, 200000)

    

def train_pinsage():
    """Train PinSage models 'manually'."""

    # precompute neighborhoods
    # train pinsage
    # save embeddings


    # Train on 3 models on 3 different node features
    features = {
        "openl3": "./dataset_final_intersect/features_openl3",
        #"vggish": "./dataset_final_intersect/features_vggish",
        #"musicnn": "./dataset_final_intersect/features_musicnn",
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
    dataset = SpotifyGraph(DATA_DIR, "./dataset_final_intersect/features_openl3")
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

    dataset = SpotifyGraph(DATA_DIR, "./dataset_final_intersect/features_openl3")
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    train_pos, test_pos = dataset.load_positives_split(os.path.join(DATA_DIR, "positives_lfm.json"))

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
        

        "PinsageBase": EmbLoader("runs/pinsage_openl3_ft/emb"),
        "PinsagePageRank": EmbLoader("runs/pinsage_pagerank_pos/emb"),

        # "PinsageRandom": EmbLoader("runs/pinsage_openl3_ft/emb"),
        # "PinsageMusicNN": EmbLoader("runs/pinsage_musicnn_ft/emb"),
        # "PinsageVggish": EmbLoader("runs/pinsage_vggish_ft/emb"),


        # "PinSageBase": PinSageWrapper(
        #     train_params={"T": 3, "lr": 0.0001, "epochs": 30, "n_layers": 2, "hard_negatives": False,
        #                     "decay": 0.95, "margin": 1e-05, "out_dim": 128},
        #     run_name="pinsage_base"
        # ),

        # "PinSageHardNeg": PinSageWrapper(
        #     train_params={"T": 3, "lr": 0.0001, "epochs": 30, "n_layers": 2, "hard_negatives": True,
        #                     "hn_min": 10, "hn_max":100, "decay": 0.95, "margin": 1e-05, "out_dim": 128},
        #     run_name="pinsage_hard_neg",
        #     log=False
        # ),

        # "PinSageFewEpochs": PinSageWrapper(
        #     train_params={"T": 3, "lr": 0.0001, "epochs": 5, "n_layers": 2, "hard_negatives": False,
        #                     "decay": 0.95, "margin": 1e-05, "out_dim": 128},
        #     run_name="pinsage_few_epochs"
        # ),

        # "PinSageManyParams": PinSageWrapper(
        #     train_params={"T": 3, "lr": 0.0001, "epochs": 30, "n_layers": 4, "hard_negatives": False,
        #                     "decay": 0.95, "margin": 1e-05, "out_dim": 256, "hidden_dim": 1024},
        #     run_name="pinsage_many_params",
        #     log=False
        # ),  

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

    
    BL_DIR = "./baselines_final_intersect"
    knn_dict = eval.get_knn_dict(baselines, g, track_ids, train_pos, test_pos, features, BL_DIR)

    results_table = eval.compute_results_table(knn_dict, test_pos, g, times=False, degree_thr=3)
    results_table.to_csv("./results_final_intersect.csv")
    print(results_table)

    beyond_table = eval.compute_beyond_accuracy_table(knn_dict, test_pos, g, features)
    beyond_table.to_csv("./results_final_intersect_beyond.csv")
    print(beyond_table)
    
    # Explore recommendation examples
    eval.crawl_embedding(knn_dict, track_ids, dataset, g, ["PinSageBase"])
    #eval.crawl_embedding(knn_dict, track_ids, dataset, g,
    #                ["PinSageBase", "OpenL3", "TrackTrackCfALS", "node2vec", "PageRank"])


if __name__ == "__main__":

    action = sys.argv[1]
    
    if action == "prepare":     # Prepare all training data (download clips, compute node features and positives)
        prepare_dataset()

    if action == "train":       # Train the PinSage models
        train_pinsage()

    if action == "eval":        # Train baselines, evaluate all models on the eval. task, compute and present results
        eval_baselines()

    if action == "all":
        prepare_dataset()
        train_pinsage()
        eval_baselines()

