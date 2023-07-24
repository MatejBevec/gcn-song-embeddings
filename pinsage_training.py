import json
import os
from os import path
from re import I

import pandas as pd
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Tanh
import time
#from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.io.image import ImageReadMode
from torch.optim.lr_scheduler import _LRScheduler

import wandb
from tqdm import tqdm


from spotify_graph import SpotifyGraph
import pinsage_model as psm

BASE_RUN_DIR = "./runs"

TEST_TRACK_INFO = None
TEST_IDS = None

def max_margin_loss(h_q, h_pos, h_neg, margin):
    norm = torch.nn.functional.normalize
    h_q, h_pos, h_neg = norm(h_q, dim=1), norm(h_pos, dim=1), norm(h_neg, dim=1)
    batch_size = h_q.shape[0]
    d = h_q.shape[1]
    q_dot_pos = torch.bmm(h_q.view(batch_size, 1, d), h_pos.view(batch_size, d, 1)).squeeze()
    q_dot_neg = torch.bmm(h_q.view(batch_size, 1, d), h_neg.view(batch_size, d, 1)).squeeze()
    dot_sum = q_dot_neg - q_dot_pos + margin
    dot_sum_zeros = torch.stack([dot_sum, torch.zeros_like(dot_sum)], 1)
    loss_vector = torch.max(dot_sum_zeros, 1, keepdim=True).values.squeeze()
    return loss_vector.mean()

def cosine_dissimilarity(a, b):
    return 1 - F.cosine_similarity(a, b)

TRIPLET_LOSS = torch.nn.TripletMarginLoss(0.0001, reduction="mean")
COSINE_TRIPLET_LOSS = torch.nn.TripletMarginWithDistanceLoss(distance_function=cosine_dissimilarity, 
                                                            margin=0.0001,
                                                            reduction="mean")

# BATCH CONSTRUCTION

def sample_positives_with_rep(positives, batch_size):
    """Sample random [batch_size] examples from the set of positive pairs.
        Returns batch = selected set of pairs, nodeset = ids of all involved nodes"""

    n_pos = positives.shape[0]
    sample = torch.randperm(n_pos)[:batch_size]
    pos_batch = positives[sample, :].to(torch.int64)
    #pos_nodeset = pos_batch.flatten().unique()

    return pos_batch

def sample_easy_negatives(all_ids, pos_batch):
    """Sample one easy negative (random node) for each positive pair 
        to make (query, positive, negative) triple"""

    n = all_ids.shape[0]
    pos_nodeset = pos_batch.flatten().unique()
    mask = torch.ones((n,)).bool()
    mask[pos_nodeset] = False
    possible_neg = all_ids[mask].to(torch.int64)
    
    negatives = possible_neg[ torch.randperm(len(possible_neg))[:pos_batch.shape[0]] ]
    batch = torch.cat((pos_batch, negatives.unsqueeze(1)), dim=1)
    nodeset = batch.flatten().unique().to(torch.int64)
    return batch, nodeset

def sample_hard_negatives(all_ids, pos_batch, nbhds, min_rank, max_rank):
    """Add one hard negative (PPR to query from min_rank to max_rank) to every positive pair."""

    queries = pos_batch[:, 0]
    rnd_ranks = torch.randint(min_rank, max_rank, (queries.shape[0],))
    hard_neg = torch.gather(nbhds[1], 1, rnd_ranks.view(-1,1)).squeeze()
    batch = torch.cat((pos_batch, hard_neg.unsqueeze(1)), dim=1)
    nodeset = batch.flatten().unique().to(torch.int64)
    return batch, nodeset

def sample_batch(all_ids, positives, batch_size, nbhds, hard_negatives=True, hn_min=10, hn_max=100):
    """Sample batch with repetition."""

    pos_batch = sample_positives_with_rep(positives, batch_size)
    if hard_negatives:
        batch, nodeset = sample_hard_negatives(all_ids, pos_batch, nbhds, hn_min, hn_max)
    else:
        batch, nodeset = sample_easy_negatives(all_ids, pos_batch)
    return batch, nodeset

def batch_variance(h):
    # to monitor potential convergence to a constant
    mean = torch.mean(h, dim=0)
    var = torch.sum( torch.pow(h-mean, 2) ) / (h.shape[0]-1)
    return torch.prod(var)

# --- TRAINING ---


class PinSage():

    """The PinSage framework. I.e. the trainer class."""

    def __init__(self, g, n_items, features, positives, log=True, load_save=True):
        
        # Name of this run
        self.run_name = "pinsage_randomft_intersect"
        # BODGE BODGE BODGE
        self.precomp_path = g.nbhds_path
        
        # --- HYPERPARAMETERS ---

        self.g = g
        self.n = n_items
        self.all_ids = torch.arange(0, n_items, 1, dtype=torch.int64)
        self.features = features
        self.positives = positives
        
        self.n_layers = 2
        self.in_dim = features.shape[1]
        self.hidden_dim = 512
        self.out_dim = 128
        self.dimensions = (self.in_dim, self.hidden_dim, self.out_dim)
        self.n_hops = 500
        self.alpha = 0.85
        self.T = 3
        self.hard_negatives = False
        self.hn_min = 10
        self.hn_max = 100

        self.nbhds = psm.precompute_neighborhoods_topt(self.g, self.n,
                                    self.n_hops, self.alpha, psm.DEF_T_PRECOMP, self.precomp_path)

        self.model = psm.PinSageModel(self.g, self.n, self.n_layers, self.dimensions,
                                    self.n_hops, self.alpha, self.T, self.nbhds)
        
        self.lr = 1e-4
        self.decay = 0.95
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.decay)
        self.margin = 1e-5 # no clue what this should be
        self.epochs = 30
        self.batch_size = 128
        self.b_per_e = 500 # batches per epoch - for now with sampling w repetition

        # --------------------


        self.embeddings = None

        run_dir = os.path.join(BASE_RUN_DIR, self.run_name)
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)
        board_dir = os.path.join(run_dir, "board")
        if not os.path.isdir(board_dir):
            os.mkdir(board_dir)

        #self.writer = SummaryWriter(board_dir)
        self.e = 0
        self.b = 0

        self.log = log
        if self.log:
            wandb.config = {"learning_rate": self.lr, "epochs": self.epochs, "batch_size": self.batch_size}
            wandb.init(project='gcn-song-embeddings', name=self.run_name)
            wandb.watch(self.model, log="all", log_freq=10, log_graph=True)
        
        self.load_save = load_save
        if self.load_save:
            self.load_model()

    
    def train_batch(self, batch):
        """Fetch and train one batch of (q, pos, neg) triples."""

        h_q = self.model(self.features, batch[:,0])
        h_pos = self.model(self.features, batch[:,1])
        h_neg = self.model(self.features, batch[:,2])

        loss = max_margin_loss(h_q, h_pos, h_neg, self.margin)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # node_feat_loss = max_margin_loss(
        #     self.features[batch[:,0], :],
        #     self.features[batch[:,1], :],
        #     self.features[batch[:,2], :],
        #     self.margin
        # )

        norm = torch.nn.functional.normalize
        node_feat_loss = COSINE_TRIPLET_LOSS(
            norm(self.features[batch[:,0], :], dim=1),
            norm(self.features[batch[:,1], :], dim=1),
            norm(self.features[batch[:,2], :], dim=1),
        )
        # node_feat_loss = COSINE_TRIPLET_LOSS(
        #     self.features[batch[:,1], :],
        #     self.features[batch[:,0], :],
        #     self.features[batch[:,2], :],
        # )        

        variance = batch_variance(h_q)

        return loss, node_feat_loss, variance

    def train(self):
        """Train the model."""

        print(f"\033[0;33mTraining PinSage...\033[0m")

        while self.e < self.epochs:
            print(f"Training epoch {self.e+1}/{self.epochs}...")
            cur_lr = self.optimizer.param_groups[0]["lr"]
            #self.writer.add_scalar("lr", cur_lr, self.e*self.b_per_e)
            t1 = time.time()

            pbar = tqdm(total=self.b_per_e)
            pbar.update(1)

            while self.b < self.b_per_e:

                batch, nodeset = sample_batch(self.all_ids, self.positives, 
                                    self.batch_size, self.nbhds, hard_negatives=self.hard_negatives,
                                    hn_min=self.hn_min, hn_max=self.hn_max)
                loss, node_feat_loss, variance = self.train_batch(batch)

                #print(f"Batch {self.b+1}/{self.b_per_e} done. Loss = {loss}")
                pbar.update(1)
                pbar.set_description(f"Loss = {loss}, bathes done")
                #self.writer.add_scalar("loss/train", loss, self.e*self.b_per_e + self.b + 1)
                if self.log:
                    wandb.log({'Train Loss': loss,
                            'Node Features Loss': node_feat_loss,
                            'Batch Variance': variance,
                            'Learning Rate': cur_lr
                            })
                
                if self.load_save:
                    self.save_model()
                self.b += 1
            
            print(f"{time.time() - t1}s elapsed.")
            pbar.close()
            self.b = 0
            self.e += 1
            self.scheduler.step()

    def embed(self, ids=None, bsize=None):
        """Return node embedding. Optionally only for 'ids' and in 'bsize' batches."""

        if ids is None:
            ids = self.all_ids
        self.model.eval()
        n = len(ids)

        if not bsize:
            self.embeddings = self.model(self.features, ids)

        else:
            self.embeddings = torch.zeros((n, self.out_dim))
            for i in range(0, n, bsize):
                ids = torch.arange(i, min(i+bsize, n))
                self.embeddings[ids, :] = self.model(self.features, ids)

        return self.embeddings

    def load_model(self):
        load_path = os.path.join(BASE_RUN_DIR, self.run_name, "state.pt")
        if os.path.isfile(load_path):

            prog = torch.load(load_path)
            self.e = prog["epochs_done"]
            self.b = prog["batches_done"]
            self.model.load_state_dict(prog["model_state"])
            self.optimizer.load_state_dict(prog["optimizer_state"])
            print(f"Loaded existing model from {load_path}.")

    def save_model(self):
        prog = {
            "epochs_done": self.e,
            "batches_done": self.b,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }
        torch.save(prog, os.path.join(BASE_RUN_DIR, self.run_name, "state.pt"))

def save_embeddings(trainer, dataset, base_run_dir=BASE_RUN_DIR, override_run_name=None):
    """Embed nodes with a 'trainer' using batching. Save embeddings."""

    track_ids = list(dataset.tracks)
    n = len(track_ids)
    bsize = 256
    run_name = override_run_name if override_run_name else trainer.run_name
    emb_dir = os.path.join(base_run_dir, run_name, "emb")
    if not os.path.isdir(emb_dir):
        os.makedirs(emb_dir)

    pbar = tqdm(total=n, desc="Saving embeddings")
    for i in range(0, n, bsize):
        #print(i, i+bsize)
        ids = torch.arange(i, min(i+bsize, n))
        #print(ids)
        #emb = trainer.embed(ids) if trainer.embeddings is None else trainer.embeddings[ids, :]
        emb = trainer.embed(ids)
        #print(emb.shape)
        for id in ids:
            str_id = track_ids[id]
            save_path = os.path.join(emb_dir, str_id + ".pt")
            if os.path.isfile(save_path):
                continue
            #print(id-i)
            torch.save(emb[id-i, :].clone().detach(), save_path)
        #print("")

        #print(f"{ids[-1]}/{n} done")
        pbar.update(bsize)
    pbar.close()


def load_embeddings(trainer, dataset, base_run_dir=BASE_RUN_DIR):
    """Load and return saved embeddings."""

    tracks = dataset.tracks
    emb_dir = os.path.join(base_run_dir, trainer.run_name, "emb")
    emb_list = []
    for track_id in tracks:
        load_path = os.path.join(emb_dir, track_id + ".pt")
        emb_list.append(torch.load(load_path))
    return torch.stack(emb_list, dim=0)


def embeddings_to_board(emb, trainer, dataset):
    """Obsolete"""
    tracks = dataset.tracks
    titles = [ f"{t['name'][:20]}\n {t['artist'][:20]}" for t in tracks.values()]
    images = []
    a = 128
    resize = torchvision.transforms.Resize((a,a))
    for track_id in tracks:
        # todo: abstract this code away in dataset class
        # todo: images are saved by album id, consider saving duplicates with track id
        image_id = tracks[track_id]["album_id"]
        load_path = os.path.join(dataset.img_dir, image_id + ".jpg") 
        if os.path.isfile(load_path):   
            # mode=ImageReadMode.RGB
            image = resize(torchvision.io.read_image(load_path))
            if image.shape[0] != 3:
                image = torch.rand((3,a,a))
        else:
            image = torch.rand((3,a,a))
        images.append(image)

    #writer = trainer.writer
    #writer.add_embedding(emb, metadata=titles, 
    #    label_img=torch.stack(images, dim=0), tag=f"pinsage:{trainer.run_name}")


def song_titles(indices, dataset, ids):
    # For error checking
    str = ""
    for j in range(indices.shape[0]):
        i = indices[j].item()
        title = dataset.tracks[ids[i]]["name"]
        title = title[0:10] if len(title) > 20 else title
        artist = dataset.tracks[ids[i]]["artist"]
        artist = artist[0:15] if len(artist) > 20 else artist
        str += f"{artist} - {title}|"
    return str

def knn_example(emb, n_examples, k, dataset, track_ids):
    # For error checking
    for ex in range(n_examples):
        query_i = torch.randint(0, emb.shape[0], (1,))
        query = emb[query_i, :]
        euclid_dist = torch.norm(emb - query, dim=1, p=None)
        knn = euclid_dist.topk(k, largest=False)
        print( song_titles(knn[1], dataset, track_ids) )
        print()


def inspect_dataset(data_dir=None, f_dir=None, pos_dir=None):
    # For error checking

    dataset = SpotifyGraph(data_dir, os.path.join(data_dir, f_dir))
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    positives = dataset.load_positives(os.path.join(data_dir, pos_dir))

    dataset2 = SpotifyGraph(data_dir, os.path.join(data_dir, f_dir))
    g2, track_ids2, col_ids2, features2 = dataset2.to_dgl_graph()
    positives2 = dataset2.load_positives(os.path.join(data_dir, pos_dir))

    e_from, e_to = g.edges()
    e_from2, e_to2 = g2.edges()
    print("\nInstanciating 2 datasets:")
    print("Graphs equal:")
    print(torch.all(e_from == e_from2))
    print(torch.all(e_to == e_to2))
    print("Tracks IDs equal: ", track_ids == track_ids2)
    print("Collection IDs equal: ", col_ids == col_ids2)
    print((features != features2).nonzero())
    print(features[0, 0:20])
    print(features2[0, 0:20])
    print("Features equal:", torch.all(features == features2))

    print("\nIndex and string ID consistency:")

    #sample = torch.randperm(len(track_ids))[0:10]
    sample = 69
    ind_nbh = g.successors(sample)
    str_nbh_from_ind = np.concatenate((track_ids,col_ids))[ind_nbh]
    # beware of this track/collection split
    str_sample = track_ids[sample]
    str_nbh = [edge["to"] for edge in dataset.graph["edges"] if edge["from"] == str_sample]
    print("IDs consistent in graph: ", str_nbh == list(str_nbh_from_ind))

    with open(os.path.join(data_dir, pos_dir), "r", encoding="utf-8") as f:
            pos_json = json.load(f)
    ind_pos = positives[sample, :]
    str_pos = [pos_json[sample]["a"], pos_json[sample]["b"]]
    str_pos_from_ind = np.concatenate((track_ids,col_ids))[ind_pos]
    print("IDs consistent in positive pairs: ", str_pos == list(str_pos_from_ind))

    print("\nSome examples:\n")
    sample = torch.randperm(positives.shape[0])[0:5]
    for ind in sample:
        pos = positives[ind, :]
        print(dataset.song_info(pos[0]))
        print(dataset.song_info(pos[1]))
        print()

    

def train_and_save(dataset, track_ids, trainer):
    trainer.train()
    save_embeddings(trainer, dataset)
    emb = load_embeddings(trainer, dataset)
    sample = torch.randperm(emb.shape[0])[:10]
    print(emb[sample,:8])
    #embeddings_to_board(emb, trainer, dataset)
    knn_example(emb, 3, 5, dataset, track_ids)


if __name__ == "__main__":

    #inspect_dataset(data_dir="./dataset_final_intersect", f_dir="features_openl3")

    dataset = SpotifyGraph("./dataset_final_intersect", "./dataset_final_intersect/features_openl3")
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    train_pos, test_pos = dataset.load_positives_split("./dataset_final_intersect/positives_lfm.json")

    trainer = PinSage(g, len(track_ids), features, train_pos, log=False, load_save=True)

    train_and_save(dataset, track_ids, trainer)
