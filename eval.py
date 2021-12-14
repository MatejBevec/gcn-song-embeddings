import os
from os import path
import time
import math

import pandas as pd
import numpy as np
import dgl
import torch
import torch.nn as nn
from torch.nn.modules.activation import Tanh
import torchvision
import matplotlib.pyplot as plt
from torchvision.io.image import ImageReadMode
from baselines import EmbLoader, Node2Vec


from spotify_graph import SpotifyGraph
import pinsage_model as psm
import pinsage_training as pst

from baselines import PredictionModel, EmbeddingModel, \
    Snore, PersPageRank, EmbLoader

PRECOMP_K = 1000
KNN_DIR = "./baselines/knn"
EMB_DIR = "./baselines/emb"

MODELS = {
    "Snore": Snore(),
    "node2vec": Node2Vec(),
    "PageRank": PersPageRank(),
    "PinSageHN": EmbLoader("runs/micro3/emb"),
    "PinSageL3": EmbLoader("runs/micro_openl3/emb"),
    "OpenL3": EmbLoader("dataset_micro/features_openl3")
}

# GOTTA TRAIN FIRST!

def precompute_model(model, model_name, g, ids, train_pos, test_pos, features):
    #emb_dir = os.path.join(EMB_DIR, model_name)
    #emb_computed = os.path.isdir(emb_dir) and len(os.listdir(emb_dir)) > 0
    knn_path = os.path.join(KNN_DIR, model_name + ".pt")
    knn_computed = os.path.isfile(knn_path)

    #todo: move knn_to_emb here and allow only loading emb.
    print()
    if not knn_computed:
        print(f"Training {model_name} model...")
        model.train(g, ids, train_pos, test_pos, features)
        if isinstance(model, EmbeddingModel):
            print("Generating and saving embeddings...")
            save_embedding(model, model_name, ids)
        print("Generating and saving knn list...")
        save_knn(model, model_name, ids)

def save_embedding(model, model_name, ids):

    save_dir = os.path.join(EMB_DIR, model_name)
    os.mkdir(save_dir) if not os.path.isdir(save_dir) else None
    all_nodes = torch.arange(0, len(ids), dtype=torch.int64) # might have to do it in batches
    
    if len(os.listdir(save_dir)) == 0:
        emb = model.embed(all_nodes)
        for i in range(emb.shape[0]):
            save_path = os.path.join(save_dir, ids[i] + ".pt")
            torch.save(emb[i,:].clone().detach(), save_path) # .clone().detach() ?

def load_embedding(model_name, ids):

    load_dir = os.path.join(EMB_DIR, model_name)
    print(f"Loading embeddings from {load_dir}...")

    if not os.path.isdir(load_dir):
        print("Load directory doesn't exist.")
        return
    fns = list(os.listdir(load_dir))
    if not len(fns) == len(ids):
        print("Number of ids and files found dont match.")
        return

    emb_list = []
    for i in range(len(ids)):
        load_path = os.path.join(load_dir, ids[i] + ".pt")
        emb_list.append(torch.load(load_path))
    
    return torch.stack(emb_list, dim=0)

def save_knn(model, model_name, ids):

    save_path = os.path.join(KNN_DIR, model_name + ".pt")
    all_nodes = torch.arange(0, len(ids), dtype=torch.int64)

    if not os.path.isfile(save_path):
        knn_w, knn_n = model.knn(all_nodes, PRECOMP_K) # (weight_mat, node_mat)
        torch.save((knn_w, knn_n), save_path)

def load_knn(model_name, ids):

    load_path = os.path.join(KNN_DIR, model_name + ".pt")
    print(f"Loading knn list from {load_path}...")
    return torch.load(load_path)

def get_knn_dict(models, g, ids, train_pos, test_pos, features):
    all_nodes = torch.arange(0, len(ids), dtype=torch.int64)
    knn_dict = {}
    
    for m_name in models:
        model = models[m_name]
        precompute_model(model, m_name, g, ids, train_pos, test_pos, features)
        knn_dict[m_name] = load_knn(m_name, ids)

    return knn_dict

# OFFLINE EVAL
def cosine_sim(a, b):
    # in: pair of batch_size * d batches of vectors
    # out: batch_size * 1 dot products of pairs of vectors 
    batch_size = a.shape[0]
    d = a.shape[1]
    q_dot_pos = torch.bmm(a.view(batch_size, 1, d), b.view(batch_size, d, 1)).squeeze()
    return q_dot_pos

def cosine_sim_single(a, b):
    # in: pair of (d) vectors
    # out: their dot product
    return torch.sum(a*b).item()

def euclid_sim_single(a, b):
    return -torch.norm(a - b, p=None).item()

def similarity_matrix(emb, sim_func):

    n = emb.shape[0]
    sim = torch.zeros((n,n))

    for i in range(n):
        for j in range(i, n):
            ij_sim = sim_func(emb[i, :], emb[j, :])
            sim[i,j], sim[j,i] = ij_sim, ij_sim
    
    return sim

#query_i = torch.randint(0, emb.shape[0], (1,))
def knn_from_emb_euclid(emb, q, K):
    # in: (n * d) embedding matrix, (d) query vector 
    euclid_dist = torch.norm(emb - q, dim=1, p=None)
    knn_dist, knn_ids = euclid_dist.topk(K, largest=False)
    return knn_ids

def knn_from_sim(sim, q, K):
    # in: (n * n) similarity matrix, (d) query vector
    sim_to_q = sim[:, q]
    knn_sim, knn_ids = sim_to_q.topk(K, largest=True)
    return knn_ids



def hit_rate(knn_mat, test_positives, K):
    # return vector/dict of hit_rate given kNN list
    n = test_positives.shape[0]
    hits = 0
    for i in range(n):
        q = test_positives[i, 0]
        pos = test_positives[i, 1]
        knn_to_q = knn_mat[q, :K]
        if pos in knn_to_q:
            hits += 1

    return hits / n

def mrr(knn_mat, test_positives, K, scaling):
    # return vector/dict of mean reciprocal rank given kNN list
    n = test_positives.shape[0]
    mrr = 0
    for i in range(n):
        q = test_positives[i, 0]
        pos = test_positives[i, 1]
        knn_to_q = knn_mat[q, :K]
        rank_pos = (knn_to_q==pos).nonzero().item() + 1 if pos in knn_to_q else K
        mrr += 1 / (rank_pos/scaling) #+1 if query is ignored
    return mrr / n

# PRESENT RESULTS
def compute_results_table(knn_dict, test_positives):

    print(test_positives)
    
    k_levels = [10, 100, 500]
    results = {}

    for model in knn_dict:
        model_results = {}
        _, knn_mat = knn_dict[model]
        max_k = knn_mat.shape[1]
        for k in k_levels:
            col_name = f"hr (k={k})"
            model_results[col_name] = hit_rate(knn_mat, test_positives, k)
        model_results["mrr"] = mrr(knn_mat, test_positives, max_k, 1)
        results[model] = model_results

    return pd.DataFrame.from_dict(results, orient="index")

# QUALITATIVE ANALYSIS

def examine_knn_weights(knn_dict):

    for m_name in knn_dict:
        print(f"{m_name}:\n")
        knn_w, _ = knn_dict[m_name]
        ranks = [0,1,2,3, 10, 50, 100, 500]
        print(knn_w[0:5, ranks])

def examine_emb(models, ids):

    for m_name in models:
        print(f"{m_name}:\n")
        emb = load_embedding(m_name, ids)
        print(emb[0:10, 0:10])

def find_knn(dataset, model_features, query, K):
    # returns K nearest neighbors to query according to given model
    pass

def print_knn(g, ids, dataset, knn_w, knn_n):
    # prints info about the query and its neighbors in a readable format

    print("\u001b[36m Nearest neighbors:")
    for i in range(0, knn_n.shape[0]):
        track = dataset.tracks[ids[knn_n[i]]]
        deg = g.in_degrees(knn_n[i]) + g.out_degrees(knn_n[i])
        print(f"{i}. [{knn_w[i]:.2f}] {track['name']} - {track['artist']} ({deg.item()})")
    print("\033[0m")
    pass

def print_query(q, ids, dataset):
    info = dataset.tracks[ids[q]]
    print("\033[0;33m", info["name"], "\033[0m")
    print(info["artist"])

def crawl_embedding(knn_dict, ids, dataset, model_names):
    # interactively crawl embedding by selecting nearest neighbors

    model_names = knn_dict.keys() if not model_names else model_names
    K = 5

    q = torch.randint(0, len(ids), (1,))
    while(True):
        print_query(q, ids, dataset)
        for m_name in model_names:
            knn_w, knn_n = knn_dict[m_name]
            print(f"{m_name}:")
            print_knn(g, ids, dataset, knn_w[q,0:K].squeeze(), knn_n[q,0:K].squeeze())
            print()
        
        choice = input("Select song or enter r for random:")
        q = torch.randint(0, len(ids), (1,))

def show_examples(dataset, model_features):
    # show a few random queries and their neighbors
    pass

def plot_tsne(dataset, model_features):
    # project embedding to tnse and visualize on a 2D plot
    pass


# TEMP
def showimg(imgs):
    side = int(math.sqrt(len(imgs))+1)
    fix, axs = plt.subplots(nrows=side, ncols=side, squeeze=False)
    for i, img in enumerate(imgs):
        img = torchvision.transforms.ToPILImage()(img.to('cpu'))
        axs[int(i/side), i%side].imshow(np.asarray(img))
        axs[int(i/side), i%side].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

if __name__ == "__main__":

    # dataset = SpotifyGraph("./dataset_mini", "./dataset_mini/features_mfcc")
    # g, track_ids, col_ids, features = dataset.to_dgl_graph()
    # pst.knn_example(features, 10, 3, dataset, track_ids)

    # img_dir = "./dataset_mini/images"
    # imgs = []
    # resize = torchvision.transforms.Resize((128,128))
    # for fn in os.listdir(img_dir)[200:500]:
    #     img = resize( torchvision.io.read_image(os.path.join(img_dir, fn), mode=ImageReadMode.RGB) )
    #     imgs.append(img)
    #     print(img.shape)
    # showimg(imgs)


    #"./dataset_mini/features_mfcc"

    positives = torch.tensor([
        [0, 1],
        [0, 5],
        [3, 4],
        [4, 2],
        [5, 6],
        [6, 7]
    ])

    knn_mat = torch.tensor([
        [0, 1, 5, 6, 7],
        [1, 0, 6, 5, 7],
        [2, 4, 3, 0, 1],
        [3, 4, 2, 7, 6],
        [4, 2, 3, 0, 1],
        [5, 6, 0, 1, 7],
        [6, 5, 7, 3, 1],
        [7, 6, 5, 0, 1],
    ])

    knn_dict = {
        "TestModel": (None, knn_mat),
        "SameTestModel": (None, knn_mat)
    }

    dataset = SpotifyGraph("./dataset_micro", None)
    g, track_ids, col_ids, _ = dataset.to_dgl_graph()
    pos = dataset.load_positives("./dataset_micro/positives.json")
    knn_dict = get_knn_dict(MODELS, g, track_ids, pos, pos, None)

    results = compute_results_table(knn_dict, pos)
    print("\n", results)

    tr_info = dataset.tracks
    _, pr_knn = knn_dict["PageRank"]
    for i in range(40, 60):
        q_id = track_ids[pos[i, 0]]
        pos_id = track_ids[pos[i, 1]]
        print(tr_info[q_id]["name"])
        print(tr_info[pos_id]["name"])
        print("-----------")
        for j in range(0, 5):
            i_id = track_ids[pr_knn[pos[i, 0], j]]
            print(tr_info[i_id]["name"])
        print()


    #examine_knn_weights(knn_dict)
    #examine_emb(["PinSage", "PinSageHN"], track_ids)
    #crawl_embedding(knn_dict, track_ids, dataset, None)


    