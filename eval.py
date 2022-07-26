import os
from os import path
import time
import math
import inspect
import shutil
import json
import re

import pandas as pd
import numpy as np
import scipy as sp
import dgl
import torch
import torch.nn as nn
from torch.nn.modules.activation import Tanh
import torchvision
import matplotlib.pyplot as plt
from torchvision.io.image import ImageReadMode
from tqdm import tqdm


from spotify_graph import SpotifyGraph
#import pinsage_model as psm
#import pinsage_training as pst

from baselines import PredictionModel, EmbeddingModel, \
    Snore, PersPageRank, EmbLoader, Random, Preferential, JaccardIndex, TrackTrackCF
from baselines import to_track_track_matrix

PRECOMP_K = 1000
BASE_DIR = "./baselines_lfm"
KNN_DIR = "./baselines_micro/knn"
EMB_DIR = "./baselines_micro/emb"

MODELS = {
    #"Snore": Snore(),
    #"Node2Vec": Node2Vec(),
    #"PageRank": PersPageRank(),
    #"Preferential": Preferential(),
    #"JaccardIndex": JaccardIndex(),
    #"PinSageMicroOpenl3": EmbLoader("runs/micro_openl3/emb"),
    #"PinSageMicroVggish": EmbLoader("runs/micro_vggish/emb"),
    "OpenL3": EmbLoader("dataset_micro/features_openl3"),
    #"Vggish": EmbLoader("dataset_micro/features_vggish2"),
    #"MusicNN": EmbLoader("dataset_micro/features_musicnn"),
    "Random": Random()
}

# GOTTA TRAIN FIRST!

def precompute_model(model, model_name, g, ids, train_pos, test_pos, features, save_dir):
    emb_dir = os.path.join(save_dir, "emb", model_name)
    emb_computed = os.path.isdir(emb_dir) and len(os.listdir(emb_dir)) > 0
    knn_path = os.path.join(save_dir, "knn", model_name + ".pt")
    knn_computed = os.path.isfile(knn_path)

    #todo: move knn_to_emb here and allow only loading emb.
    print()
    train_time = 0
    emb_time = 0
    if not knn_computed:
        print(f"Training {model_name} model...")
        t0 = time.time()
        model.train(g, ids, train_pos, test_pos, features)
        train_time = time.time() - t0
        if isinstance(model, EmbeddingModel):
            print("Generating and saving embeddings...")
            emb_time = save_embedding(model, model_name, ids, save_dir)
        print("Generating and saving knn list...")
        save_knn(model, model_name, ids, save_dir, train_time=train_time, emb_time=emb_time)


def save_embedding(model, model_name, ids, save_dir):

    save_dir = os.path.join(save_dir, "emb", model_name)
    os.makedirs(save_dir) if not os.path.isdir(save_dir) else None
    all_nodes = torch.arange(0, len(ids), dtype=torch.int64) # might have to do it in batches
    
    emb_time = 0
    if len(os.listdir(save_dir)) == 0:
        t0 = time.time()
        emb = model.embed(all_nodes)
        emb_time = time.time() - t0
        for i in range(emb.shape[0]):
            save_path = os.path.join(save_dir, ids[i] + ".pt")
    
            torch.save(emb[i,:].clone().detach(), save_path)
    
    return emb_time

def load_embedding(model_name, ids, save_dir):

    load_dir = os.path.join(save_dir, "emb", model_name)
    print(f"Loading embeddings from {load_dir}...")

    if not os.path.isdir(load_dir):
        print("Load directory doesn't exist.")
        return
    fns = list(os.listdir(load_dir))
    if not len(fns) == len(ids):
        print("Number of ids and files found don't match.")
        return

    emb_list = []
    for i in range(len(ids)):
        load_path = os.path.join(load_dir, ids[i] + ".pt")
        emb_list.append(torch.load(load_path))
    
    return torch.stack(emb_list, dim=0)

def save_knn(model, model_name, ids, save_dir, train_time=0, emb_time=0):

    save_dir = os.path.join(save_dir, "knn")
    os.makedirs(save_dir) if not os.path.isdir(save_dir) else None
    save_path = os.path.join(save_dir, model_name + ".pt")
    all_nodes = torch.arange(0, len(ids), dtype=torch.int64)

    knn_time = 0
    if not os.path.isfile(save_path):
        # compute knn in batches due to memory constraints
        b_size = 1000
        n = len(all_nodes)
        knn_w_list = []
        knn_n_list = []
        pbar = tqdm(total=n, desc="Computing KNN")
        for i in range(0, n, b_size):
            end = min(i+b_size, n)
            t0 = time.time()
            b_knn_w, b_knn_n = model.knn(all_nodes[i:end], PRECOMP_K) # (weight_mat, node_mat)
            knn_w_list.append(b_knn_w)
            knn_n_list.append(b_knn_n)
            knn_time += time.time() - t0
            #print(f"knn: {end}/{n} done")
            pbar.update(b_size)
        torch.save((torch.cat(knn_w_list, dim=0), 
                    torch.cat(knn_n_list, dim=0),
                    train_time,
                    emb_time, # BODGE: save train, emb., and knn constr. times together with knn list
                    knn_time
                    ),
                save_path)
        pbar.close()


def load_knn(model_name, ids, save_dir):

    load_path = os.path.join(save_dir, "knn", model_name + ".pt")
    print(f"Loading knn list from {load_path}...")
    return torch.load(load_path)

# def get_knn_dict(models, g, ids, train_pos, test_pos, features, save_dir):
#     # returns {"model_name": (knn_weights, knn_indices)} for all models
#     # knn_weights, knn_indices are of shape n*k

#     all_nodes = torch.arange(0, len(ids), dtype=torch.int64)
#     knn_dict = {}
    
#     for m_name in models:
#         model = models[m_name]
#         precompute_model(model, m_name, g, ids, train_pos, test_pos, features, save_dir)
#         knn_dict[m_name] = load_knn(m_name, ids, save_dir)

#     return knn_dict

def get_knn_dict(models, g, ids, train_pos, test_pos, features, save_dir):

    all_nodes = torch.arange(0, len(ids), dtype=torch.int64)
    
    for m_name in models:
        model = models[m_name]
        precompute_model(model, m_name, g, ids, train_pos, test_pos, features, save_dir)

    knn_dict = LazyKnnDict(list(models.keys()), all_nodes, save_dir)
    return knn_dict

class LazyKnnDict():
    
    def __init__(self, model_names, ids, save_dir):
        self.all_nodes = torch.arange(0, len(ids), dtype=torch.int64)
        self.models = model_names
        self.save_dir = save_dir
        self.idx = 0
        self.times = {m_name: None for m_name in model_names}
    
    def __getitem__(self, m_name):
        tup = load_knn(m_name, self.all_nodes, self.save_dir)
        knn_w, knn_n = tup[0], tup[1].to(torch.int64)
        #print(knn_n[:10, :3])
        return knn_w, knn_n

    # BODGE BODGE BODGE
    # TODO: REFACTOR THIS ATROCITY
    def get_times(self, m_name):
        if not self.times[m_name]:
            tpl = load_knn(m_name, self.all_nodes, self.save_dir)
            if len(tpl) == 5:
                _, _, traint, embt, knnt = tpl
            else:
                traint, embt, knnt = 0, 0, 0
            self.times[m_name] = (traint, embt, knnt)
        return self.times[m_name]

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.__len__():
            raise StopIteration
        self.idx += 1
        return self.models[self.idx-1]





# OFFLINE EVAL
# -----------------------------------------------------

# def cosine_sim(a, b):
#     # in: pair of batch_size * d batches of vectors
#     # out: batch_size * 1 dot products of pairs of vectors 
#     batch_size = a.shape[0]
#     d = a.shape[1]
#     q_dot_pos = torch.bmm(a.view(batch_size, 1, d), b.view(batch_size, d, 1)).squeeze()
#     return q_dot_pos

# def cosine_sim_single(a, b):
#     # in: pair of (d) vectors
#     # out: their dot product
#     return torch.sum(a*b).item()

# def euclid_sim_single(a, b):
#     return -torch.norm(a - b, p=None).item()

# def similarity_matrix(emb, sim_func):

#     n = emb.shape[0]
#     sim = torch.zeros((n,n))

#     for i in range(n):
#         for j in range(i, n):
#             ij_sim = sim_func(emb[i, :], emb[j, :])
#             sim[i,j], sim[j,i] = ij_sim, ij_sim
    
#     return sim

# #query_i = torch.randint(0, emb.shape[0], (1,))
# def knn_from_emb_euclid(emb, q, K):
#     # in: (n * d) embedding matrix, (d) query vector 
#     euclid_dist = torch.norm(emb - q, dim=1, p=None)
#     knn_dist, knn_ids = euclid_dist.topk(K, largest=False)
#     return knn_ids

# def knn_from_sim(sim, q, K):
#     # in: (n * n) similarity matrix, (d) query vector
#     sim_to_q = sim[:, q]
#     knn_sim, knn_ids = sim_to_q.topk(K, largest=True)
#     return knn_ids


# ACCURACY

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

def mrr(knn_mat, test_positives, K, scaling=1):
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


# BEYOND ACCURACY

def cosine_sim_mat(batch, eps=1e-10):
    # given a batch of vectors (n*d), return n*n matrix of pairwise similaries
    n, d = batch.shape
    dot_prod = torch.mm( batch, batch.transpose(1, 0) )
    lengths = torch.norm(batch, dim=1)
    lengths_mat = lengths.unsqueeze(1).repeat((1, n))
    lengths_mul = lengths_mat * lengths_mat.transpose(1, 0)

    cosine_sim = dot_prod / lengths_mul
    return cosine_sim

def cosine_sim_mat_sparse(csr_batch, eps=1e-10):
    # given a batch of vectors (n*d) as csr_matrix, return n*n
    pass


def intra_diversity(knn_mat, test_positives, K, features):
    # return vector of intra-list diversity (1 - cosine similarity) with respect to "features"
    # n - number of queries in knn_mat, N - number of all nodes
    # features - node feature matrix of shape N*d

    n = knn_mat.shape[0]
    all_sims = torch.zeros(n)
    for query in range(n):
        rec = knn_mat[query, :K].long()
        rec_features = features[rec, :]
        sim_mat = cosine_sim_mat(rec_features)
        avg_sim = torch.mean(sim_mat)
        all_sims[query] = avg_sim
    
    all_avg_sims = torch.mean(all_sims).item()
    return 1 - all_avg_sims

def inter_diversity(knn_mat, test_positives, K, N, n_pairs=10000):
    # return inter-list diversity (personalization) with respect to indices? over all recommendations
    # n - number of queries in knn_mat, N - number of all nodes

    n = knn_mat.shape[0]
    one_hot_mat = sp.sparse.lil_matrix((n, N))
    one_hot_list = []
    for i in range(n):
        selection = knn_mat[i, :K]

        one_hot_mat[i, selection] = 1

    one_hot_mat = one_hot_mat.tocsr()

    # Compute cosine similarity between [n_pairs] random pairs of recommendation sets
    distances = np.ndarray(n_pairs)
    rnd = np.random.randint(0, n, (n_pairs, 2))
    for i in range(0, n_pairs):
        v1 = one_hot_mat[rnd[i, 0], :].toarray().squeeze()
        v2 = one_hot_mat[rnd[i, 1], :].toarray().squeeze()
        cos = sp.spatial.distance.cosine(v1, v2)
        distances[i] = cos

    avg_dist = np.mean(distances)
    return avg_dist

def inter_diversity_matrix(knn_mat, test_positives, K, N, n_examples=10000):
    # return inter-list diversity (personalization) with respect to indices? over all recommendations
    # n - number of queries in knn_mat, N - number of all nodes

    n = knn_mat.shape[0]
    one_hot_mat = sp.sparse.lil_matrix((n, N))
    one_hot_list = []
    for i in range(n):
        selection = knn_mat[i, :K]
        one_hot_mat[i, selection] = 1

    one_hot_mat = one_hot_mat.tocsr()

    # Compute cosine similarity between [n_pairs] random pairs of recommendation sets
    sim_list = []
    rnd = np.random.randint(0, n, (n_examples))
    bsize = 512
    for i in range(0, n_examples, bsize):
        n = min(bsize, n-(i+bsize))
        batch = torch.from_numpy( one_hot_mat[rnd[i:i+n], :].toarray() )
        sim_mat = cosine_sim_mat(batch)
        sim_list.append(sim_mat.flatten())

    #sim_mat = cosine_sim_mat(one_hot)
    #avg_sim = torch.mean(sim_mat)
    avg_sim = torch.mean(torch.cat(sim_list, dim=0)).item()
    return 1 - avg_sim

def coverage(knn_mat, test_positives, K=500, all_nodes=True, queries=None):
    # return coverage over all recommendations
    # if all_node is True use all graph nodes as queries, else only use nodes in test_positives
    #print("t0")
    recs = torch.flatten(knn_mat[:, 1:K+1]).long() if all_nodes else torch.flatten(test_positives.long())
    #rec_nodes = set(recs)
    rec_nodes = torch.unique(recs)
    #print("rec nodes done")
    all_nodes = torch.arange(0, knn_mat.shape[0], dtype=torch.int64) if not queries else torch.unique(queries)
    
    coverage = rec_nodes.shape[0] / all_nodes.shape[0]
    #print("intersect done")

    return coverage

def average_degree(knn_mat, g, test_positives, K, queries=None):
    # return vector of average graph degree (novelty) of generated recommendations

    queries = torch.arange(0, knn_mat.shape[0], dtype=torch.int64) if not queries else queries
    rec_nodes = torch.flatten(knn_mat[:, :K]).long()
    degrees = g.in_degrees(rec_nodes)

    return torch.mean(degrees.double()).item()

def degree_dist(knn_mat, g, test_positives, K, queries=None):
    # return degree distribution (counts) over all recommendations

    queries = torch.arange(0, knn_mat.shape[0], dtype=torch.int64) if not queries else queries
    rec_nodes = torch.flatten(knn_mat[:, :K]).long()
    degrees = g.in_degrees(rec_nodes)

    degree_dist = torch.unique(degrees, sorted=True, return_counts=True)
    return degree_dist

def low_degree_accuracy(knn_mat, g, test_positives, K, degree_thr, acc_func, queries=None, track_ids=None):
    # generate recommendation, keep only queries with degree under "degree_thr"
    # compute "acc_fun" on these low-degree queries
    queries = torch.arange(0, knn_mat.shape[0], dtype=torch.int64) if not queries else queries
    sel_indices = queries[g.in_degrees(queries) <= degree_thr]

    pos_selection = np.array([test_positives[i, 0] in sel_indices for i in range(test_positives.shape[0])])
    low_deg_positives = test_positives[pos_selection, :]

    score = acc_func(knn_mat, low_deg_positives, K)
    #score = hit_rate(knn_mat, low_deg_positives, 500)
    return score

def low_co_accuracy(knn_mat, g, test_positives, K, co_thr, acc_func, queries=None, track_ids=None):
    # compute "acc_fun" on queries with under co_thr track-track co-occurences
    queries = torch.arange(0, knn_mat.shape[0], dtype=torch.int64) if not queries else queries
    ids = np.arange(0, knn_mat.shape[0])
    ttmat = to_track_track_matrix(ids, test_positives)
    co_counts = torch.from_numpy(np.sum(ttmat, axis=1)).squeeze()[queries]

    sel_indices = queries[co_counts <= co_thr]
    sel_indices = set(sel_indices.tolist())
    
    query_list = test_positives[:, 0].squeeze().tolist()
    sel_positives = np.array([q in sel_indices for q in query_list])
    low_co_positives = test_positives[sel_positives, :]

    score = acc_func(knn_mat, low_co_positives, K)
    return score




# PRESENT RESULTS

def compute_results_table(knn_dict, test_positives, g, times=True):
    
    k_levels = [10, 100, 500]
    results = {}

    for model in knn_dict:
        model_results = {}
        _, knn_mat = knn_dict[model]
        max_k = knn_mat.shape[1]

        for k in k_levels:
            col_name = f"hr (k={k})"
            model_results[col_name] = hit_rate(knn_mat, test_positives, k)

        model_results["mrr"] = mrr(knn_mat, test_positives, 1000, 1)

        model_results["low-degree accuracy"] = low_degree_accuracy(
            knn_mat, g, test_positives, 1000, degree_thr=1, acc_func=mrr)
        results[model] = model_results

        model_results["low-co accuracy"] = low_co_accuracy(
            knn_mat, g, test_positives, 1000, co_thr=1, acc_func=mrr)
        results[model] = model_results

        if times:
            traint, embt, knnt = knn_dict.get_times(model)
            model_results["t (train)"] = traint
            model_results["t (emb)"] = embt
            model_results["t (knn)"] = knnt

    return pd.DataFrame.from_dict(results, orient="index")

def compute_beyond_accuracy_table(knn_dict, test_positives, g, features):

    k = 100
    results = {}

    for model in knn_dict:
        model_results = {}
        _, knn_mat = knn_dict[model]
        max_k = knn_mat.shape[1]

        print(f"Computing beyond-accuracy metrics for {model}")

        model_results = {
            "intra diversity": intra_diversity(knn_mat, test_positives, k, features),
            "inter diversity": inter_diversity(knn_mat, test_positives, k, features.shape[0]),
            "coverage": coverage(knn_mat, test_positives, K=100),
            "average degree": average_degree(knn_mat, g, test_positives, k),
            #"low-degree accuracy": low_degree_accuracy(knn_mat, g, test_positives, 100, degree_thr=1, acc_func=hit_rate)        
        }
        results[model] = model_results

    print(results)
    return pd.DataFrame.from_dict(results, orient="index")



# QUALITATIVE ANALYSIS

def examine_knn_weights(knn_dict):

    for m_name in knn_dict:
        print(f"{m_name}:\n")
        knn_w, _ = knn_dict[m_name]
        ranks = [0,1,2,3, 10, 50, 100, 500]
        print(knn_w[0:10, ranks])

def examine_emb(models, ids, save_dir):

    for m_name in models:
        print(f"{m_name}:\n")
        emb = load_embedding(m_name, ids, save_dir)
        print(emb[0:10, 0:10])

def find_knn(dataset, model_features, query, K):
    # returns K nearest neighbors to query according to given model
    pass

def print_knn(g, ids, dataset, knn_w, knn_n):
    # prints info about the query and its neighbors in a readable format

    print(knn_n)
    print(type(knn_n))
    

    print("\u001b[36m Nearest neighbors:")
    for i in range(0, knn_n.shape[0]):
        track = dataset.tracks[ids[knn_n[i]]]
        deg = g.in_degrees(knn_n[i]) + g.out_degrees(knn_n[i])
        print(f"{i}. [{knn_w[i]:.3f}] {track['name']} - {track['artist']} ({deg.item()})")
    print("\033[0m")
    pass

def print_query(q, ids, dataset):
    info = dataset.tracks[ids[q]]
    print("\033[0;33m", info["name"], "\033[0m")
    print(info["artist"])

def crawl_embedding(knn_dict, ids, dataset, g, model_names=None):
    # interactively crawl embedding by selecting random queries

    model_names = knn_dict.keys() if not model_names else model_names
    K = 10

    q = torch.randint(0, len(ids), (1,))
    while(True):
        knn_lists = []
        print_query(q, ids, dataset)
        for i,m_name in enumerate(model_names):
            knn_w, knn_n = knn_dict[m_name]
            knn_lists.append(knn_n)
            print(f"[{i}]{m_name}:")
            print_knn(g, ids, dataset, knn_w[q,0:K].squeeze(), knn_n[q,0:K].squeeze())
            print()
        
        choice = input("Select song or enter r for random:")
        print(choice)
        if choice == "e":
            for i,m_name in enumerate(model_names):
                export_recommendation_list(g, ids, dataset, q, knn_lists[i], m_name)
            export_recommendation_figure(g, ids, dataset, q, knn_dict, model_names)
        q = torch.randint(0, len(ids), (1,))


def export_recommendation_lists(g, ids, dataset, queries, knn_dict, model_names=None):



    model_names = knn_dict.keys() if not model_names else model_names
    for q in queries:
        for m_name in model_names:
            knn_w, knn_n = knn_dict[m_name]
            q_index = ids.index(q)
            export_recommendation_list(g, ids, dataset, q_index, knn_n, m_name)
        export_recommendation_figure(g, ids, dataset, q_index, knn_dict, model_names)


def export_recommendation_list(g, ids, dataset, q, knn_n, m_name, k=5):
    
    q = q.item() if isinstance(q, torch.Tensor) else q
    rec_list = [q]
    rec_list.extend(knn_n[q, :k].squeeze().tolist())
    print(rec_list)
    info_list = []
    dir_name = os.path.join("examples", dataset.tracks[ids[q]]['name'], m_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    for i,tr in enumerate(rec_list):
        track_info = dataset.tracks[ids[tr]]
        info_list.append({
            "title": track_info["name"],
            "artist": track_info["artist"],
            "album": track_info["album"]
        })
        export_track_image(dataset.base_dir, dir_name, track_info, i)

    with open(os.path.join(dir_name, "list.json"), "w", encoding="utf-8") as f:
        json.dump(info_list, f, indent=2)



def export_track_image(dataset_dir, save_dir, track_info, rank):
    prefix = f"[{'q' if rank == 0 else rank}]"
    name = track_info["name"]
    name = re.sub('[/]', '', name)
    album_id = track_info["album_id"]
    src_path = os.path.join(dataset_dir, "images", album_id + ".jpg")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dest_path = os.path.join(save_dir, name + ".jpg")
    shutil.copy(src_path, dest_path)


def export_recommendation_figure(g, ids, dataset, q, knn_dict, model_names, k=4):

    with open("examples_template.tex", "r", encoding="utf-8") as f:
        template = f.read()

    q = q.item() if isinstance(q, torch.Tensor) else q
    dir_name = os.path.join("examples",dataset.tracks[ids[q]]['name'])
    fig_path = os.path.join(dir_name, "figure.tex")
    print(fig_path)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    
    for m,m_name in enumerate(model_names):

        # GET SONG INFO
        _, knn_n = knn_dict[m_name]
        rec_list = [q]
        rec_list.extend(knn_n[q, :k].squeeze().tolist())
        print(rec_list)
        info_list = []

        template = template.replace(f"<method_{m}>", m_name)

        for i,tr in enumerate(rec_list):
            track_info = dataset.tracks[ids[tr]]
            info_list.append({
                "title": track_info["name"],
                "artist": track_info["artist"],
                "album": track_info["album"]
            })
            cover_path = os.path.join(dir_name, "covers", track_info["name"] + ".jpg")
            export_track_image(dataset.base_dir, os.path.join(dir_name, "covers"), track_info, i)
        
            template = template.replace(f"<cover_{m}_{i}>", cover_path)
            template = template.replace(f"<title_{m}_{i}>", track_info["name"])
            template = template.replace(f"<artist_{m}_{i}>", track_info["artist"])
            template = template.replace(f"<album_{m}_{i}>", track_info["album"])

    with open(fig_path, "w", encoding="utf-8") as f:
        f.write(template)  



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
    # for fn in os.listdir(img_dir)[200:1000]:
    #     img = resize( torchvision.io.read_image(os.path.join(img_dir, fn), mode=ImageReadMode.RGB) )
    #     imgs.append(img)
    #     print(img.shape)
    # showimg(imgs)

    # positives = torch.tensor([
    #     [0, 1],
    #     [0, 5],
    #     [3, 4],
    #     [4, 2],
    #     [5, 6],
    #     [6, 7]
    # ])

    # knn_mat = torch.tensor([
    #     [0, 1, 5, 6, 7],
    #     [1, 0, 6, 5, 7],
    #     [2, 4, 3, 0, 1],
    #     [3, 4, 2, 7, 6],
    #     [4, 2, 3, 0, 1],
    #     [5, 6, 0, 1, 7],
    #     [6, 5, 7, 3, 1],
    #     [7, 6, 5, 0, 1],
    # ])

    # knn_dict = {
    #     "TestModel": (None, knn_mat),
    #     "SameTestModel": (None, knn_mat)
    # }

    dataset = SpotifyGraph("./dataset_final_intersect", None)#"./dataset_small/features_openl3")
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    pos = dataset.load_positives("./dataset_final_intersect/positives_lfm.json")
    train_pos, test_pos = dataset.load_positives_split("./dataset_final_intersect/positives_lfm.json")

    #models =  {"OpenL3": EmbLoader("dataset_small/features_openl3")}
    #models =  {"PinSageOpenL3LFMBestBest": EmbLoader("runs_gs4/gridsearch#0.0.0.0.0.1.0.0/emb")}
    models =  {"TrackTrackCfALS": TrackTrackCF()}#, "PageRank": PersPageRank()}
    knn_dict = get_knn_dict(models, g, track_ids, train_pos, test_pos, None, "./baselines_final_intersect")
    #_, knn_mat = knn_dict["TrackTrackCfALS"]
    #_, knn_mat = knn_dict["PageRank"]
    _, knn_mat = knn_dict["ColTrackCfALS"]


    # all_indices = torch.arange(0, len(track_ids))
    # print(all_indices[:10])
    # sel_indices = all_indices[g.in_degrees(all_indices) <= 1]
    # print(sel_indices[:10])
    # print(len(sel_indices))
    # sel = np.array([pos[i, 0] in sel_indices for i in range(pos.shape[0])])
    # print(sel[:10])
    # indices_in_pos = np.unique(pos[sel, 0])
    # print(indices_in_pos[:10])
    # print(len(indices_in_pos))

    track_ids = np.array(track_ids)

    score = mrr(knn_mat, test_pos, 1000)
    print("\n normal mrr:")
    print(score)
    low_deg_score = low_degree_accuracy(knn_mat, g, test_pos, 1000, 1, mrr, track_ids=track_ids)
    print("\n low degree mrr:")
    print(low_deg_score)
    low_co_score = low_co_accuracy(knn_mat, g, pos, 1000, 1, mrr, track_ids=track_ids)
    print("\n low co mrr")
    print(low_co_score)


    score = hit_rate(knn_mat, test_pos, 100)
    print("\n normal hitrate:")
    print(score)
    low_deg_score = low_degree_accuracy(knn_mat, g, test_pos, 100, 1, hit_rate, track_ids=track_ids)
    print("\n low degree hitrate:")
    print(low_deg_score)
    low_co_score = low_co_accuracy(knn_mat, g, pos, 100, 1, hit_rate, track_ids=track_ids)
    print("\n low co hitrate")
    print(low_co_score)


    # results = compute_results_table(knn_dict, pos, g)
    # print("\n", results)
    # results.to_csv("results.csv")

    #results = compute_beyond_accuracy_table(knn_dict, pos, g, features)
    #print(results)

    #examine_knn_weights(knn_dict)
    #examine_emb(["PinSage", "PinSageHN"], track_ids)
    #crawl_embedding(knn_dict, track_ids, dataset, g, ["PageRank", "PinSageBias", "OpenL3"])


    