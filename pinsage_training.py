import os
from os import path

import pandas as pd
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Tanh
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.io.image import ImageReadMode


from spotify_graph import SpotifyGraph
import pinsage_model as psm

BASE_RUN_DIR = "./runs"
PRECOMP_NAME = "neighborhoods_micro.pt"

TEST_TRACK_INFO = None
TEST_IDS = None

def max_margin_loss(h_q, h_pos, h_neg, margin):
    batch_size = h_q.shape[0]
    d = h_q.shape[1]
    q_dot_pos = torch.bmm(h_q.view(batch_size, 1, d), h_pos.view(batch_size, d, 1)).squeeze()
    q_dot_neg = torch.bmm(h_q.view(batch_size, 1, d), h_neg.view(batch_size, d, 1)).squeeze()
    dot_sum = q_dot_neg - q_dot_pos + margin
    dot_sum_zeros = torch.stack([dot_sum, torch.zeros_like(dot_sum)], 1)
    loss_vector = torch.max(dot_sum_zeros, 1, keepdim=True).values.squeeze()
    return loss_vector.mean()


# BATCH CONSTRUCTION

def sample_positives_with_rep(positives, batch_size):
    # sample random [batch_size] examples from the set of positive pairs
    # returns batch = selected set of pairs, nodeset = ids of all involved nodes

    n_pos = positives.shape[0]
    sample = torch.randperm(n_pos)[:batch_size]
    pos_batch = positives[sample, :].to(torch.int64)
    #pos_nodeset = pos_batch.flatten().unique()

    return pos_batch

def sample_easy_negatives(all_ids, pos_batch):
    # sample one easy negative (random node) for each positive pair
    # to make (query, positive, negative) triple
    # probably not what they do in the paper, but first attempt
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
    # sample one hard negative (PPR to query from min_rank to max_rank) for every positive pair
    # TODO: test this
    queries = pos_batch[:, 0]
    rnd_ranks = torch.randint(min_rank, max_rank, (queries.shape[0],))
    #print(rnd_ranks[0:10])
    hard_neg = torch.gather(nbhds[1], 1, rnd_ranks.view(-1,1)).squeeze()
    #print(hard_neg[0:10])
    batch = torch.cat((pos_batch, hard_neg.unsqueeze(1)), dim=1)
    nodeset = batch.flatten().unique().to(torch.int64)
    return batch, nodeset

def sample_batch(all_ids, positives, batch_size, nbhds):
    # sample batch with repetition and with random negative per positive pair
    pos_batch = sample_positives_with_rep(positives, batch_size)
    #batch, nodeset = sample_easy_negatives(all_ids, pos_batch)
    batch, nodeset = sample_hard_negatives(all_ids, pos_batch, nbhds, 10, 100)
    return batch, nodeset

def batch_variance(h):
    # to monitor potential convergence to a constant
    mean = torch.mean(h, dim=0)
    var = torch.sum( torch.pow(h-mean, 2) ) / (h.shape[0]-1)
    return torch.prod(var)

# TRAINING

class PinSage():

    def __init__(self, g, n_items, features, positives):

        #todo: somehow wrap the parameters in a dict

        self.run_name = "high_lr_standardized_hn4"
        self.precomp_path = PRECOMP_NAME

        self.g = g
        self.n = n_items
        self.all_ids = torch.arange(0, n_items, 1, dtype=torch.int64)
        self.features = features
        self.positives = positives
        
        self.n_layers = 2
        self.in_dim = features.shape[1]
        self.dimensions = (self.in_dim, 512, 128) # (in/features, hidden, out/embedding)
        self.n_hops = 500
        self.alpha = 0.85
        self.T = 3

        self.nbhds = psm.precompute_neighborhoods_topt(self.g, self.n,
                                    self.n_hops, self.alpha, psm.DEF_T_PRECOMP, self.precomp_path)

        self.model = psm.PinSageModel(self.g, self.n, self.n_layers, self.dimensions,
                                    self.n_hops, self.alpha, self.T, self.nbhds)
        
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.90)
        self.margin = 1e-5 # no clue what this should be
        self.epochs = 10
        self.batch_size = 128
        self.b_per_e = 50 # batches per epoch - for now with sampling w repetition

        # BUG: loss becomes NaN after 1st batch if i change T or n_layers

        self.embeddings = None

        run_dir = os.path.join(BASE_RUN_DIR, self.run_name)
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)
        board_dir = os.path.join(run_dir, "board")
        if not os.path.isdir(board_dir):
            os.mkdir(board_dir)

        self.writer = SummaryWriter(board_dir)
        self.e = 0
        self.b = 0
        
        self.load_model()

    
    def train_batch(self, batch):
        # batch = batch of triples (q, pos, neg)

        h_q = self.model(self.features, batch[:,0])
        h_pos = self.model(self.features, batch[:,1])
        h_neg = self.model(self.features, batch[:,2])
        # -> check if node order is preserved when getting back the embedding

        # print(batch[:,0])
        # print(batch[:,1])
        # print(batch[:,2])

        # print(h_q[1,0:4])
        # print(h_pos[1,0:4])
        # print(h_neg[1,0:4])

        loss = max_margin_loss(h_q, h_pos, h_neg, self.margin)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        variance = batch_variance(h_q)
        print(f"Batch variance = {variance}")

        return loss

    def train(self):
        # another option: data gets passed here not at init
        # todo: train test split

        while self.e < self.epochs:
            print(f"Training epoch {self.e+1}/{self.epochs}...")
            cur_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("lr", cur_lr, self.e*self.b_per_e)
            t1 = time.time()

            while self.b < self.b_per_e:

                batch, nodeset = sample_batch(self.all_ids, self.positives, self.batch_size, self.nbhds)
                loss = self.train_batch(batch)

                print(f"Batch {self.b+1}/{self.b_per_e} done. Loss = {loss}")
                self.writer.add_scalar("loss/train", loss, self.e*self.b_per_e + self.b + 1)
                
                self.save_model()
                self.b += 1
            
            print(f"{time.time() - t1}s elapsed.")
            self.b = 0
            self.e += 1
            self.scheduler.step()

    def embed(self):

        self.model.eval()
        self.embeddings = self.model(self.features, self.all_ids)
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


# TODO: identical interface to baseline methods
def save_embeddings(trainer, dataset):
    tracks = dataset.tracks
    emb = trainer.embeddings if trainer.embeddings != None else trainer.embed()
    emb_dir = os.path.join(BASE_RUN_DIR, trainer.run_name, "emb")
    if not os.path.isdir(emb_dir):
        os.mkdir(emb_dir)
    for i in range(emb.shape[0]):
        track_id = list(tracks)[i]
        save_path = os.path.join(emb_dir, track_id + ".pt")
        if os.path.isfile(save_path):
            continue
        torch.save(emb[i, :].clone().detach(), save_path)

def load_embeddings(trainer, dataset):
    tracks = dataset.tracks
    emb_dir = os.path.join(BASE_RUN_DIR, trainer.run_name, "emb")
    emb_list = []
    for track_id in tracks:
        load_path = os.path.join(emb_dir, track_id + ".pt")
        emb_list.append(torch.load(load_path))
    return torch.stack(emb_list, dim=0)

def embeddings_to_board(emb, trainer, dataset):
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

    writer = trainer.writer
    writer.add_embedding(emb, metadata=titles, 
        label_img=torch.stack(images, dim=0), tag=f"pinsage:{trainer.run_name}")


# BASELINE: small FC NN, trained with the same positives and with node feature (content embedding) input
class FineTunedNNModel(nn.Module):
    def __init__(self, in_dim):
        super(FineTunedNNModel, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = 512
        self.out_dim = 128

        self.FC1 = nn.Linear(self.in_dim, self.hid_dim)
        self.FC2 = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, x):
        x = F.relu(self.FC1(x))
        y = self.FC2(x)
        return y

class FineTunedNN():
    def __init__(self):
        self.model = None
        pass

    def train_batch(self, batch):
        # batch are triples of form (query, positive, negative) -> ids
        #print(batch)
        h_q = self.model(self.features[batch[:,0],:])
        h_pos = self.model(self.features[batch[:,1],:])
        h_neg = self.model(self.features[batch[:,2],:])

       

        loss = max_margin_loss(h_q, h_pos, h_neg, self.margin)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def train(self, ids, positives, features, nbhds):
        self.features = features
        self.n = self.features.shape[0]
        self.ids = ids
        self.model = FineTunedNNModel(features.shape[1])
        self.lr = 1e-4
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.90)
        self.margin = 1e-5
        self.nbhds = nbhds

        epochs = 100
        batch_size = 128
        b_per_e = 100
        
        for e in range(epochs):
            print(f"Epoch {e}:")
            #for i in range(0, self.n, batch_size):
                #end = min(i+batch_size, self.n)
            for i in range(0, b_per_e):
                batch = sample_batch(ids, positives, batch_size, self.nbhds)
                
                loss = self.train_batch(batch[0])

                if i%20 == 0:
                    print(f"{i}/{b_per_e} batches done. Loss = {loss}")
            self.scheduler.step()


    def embed(self, nodeset):
        pass
    
    def knn(self, nodeset, k):
        pass



def song_titles(indices, dataset, ids):
    # get song title, artist from node index
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
    for ex in range(n_examples):
        query_i = torch.randint(0, emb.shape[0], (1,))
        query = emb[query_i, :]
        euclid_dist = torch.norm(emb - query, dim=1, p=None)
        knn = euclid_dist.topk(k, largest=False)
        print( song_titles(knn[1], dataset, track_ids) )
        print()

    

if __name__ == "__main__":

    dataset = SpotifyGraph("./dataset_micro", "./dataset_micro/features_openl3")
    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    positives = dataset.load_positives("./dataset_micro/positives.json")

    trainer = PinSage(g, len(track_ids), features, positives)

    print(features.shape)
    print(features[0])

    #torch.autograd.set_detect_anomaly(True)
    # trainer.train()
    # save_embeddings(trainer, dataset)
    # emb = load_embeddings(trainer, dataset)
    # sample = torch.randperm(emb.shape[0])[:10]
    # print(emb[sample,:8])
    # #embeddings_to_board(emb, trainer, dataset)
    # knn_example(emb, 3, 5, dataset, track_ids)

    fcn = FineTunedNN()
    fcn.train(track_ids, positives, features, trainer.nbhds)


    # TEST_TRACK_INFO = dataset.tracks
    # TEST_IDS = track_ids
    # nbhds = trainer.nbhds
    # pos = sample_positives_with_rep(positives, 8)
    # all_nodes = torch.arange(0, len(track_ids)).long()
    # #batch, nodeset = sample_hard_negatives(all_nodes, pos, nbhds, 10, 100)
    # batch, nodeset = sample_easy_negatives(all_nodes, pos)
    # print(batch)
    # for i in range(batch.shape[0]):
    #     id1 = TEST_IDS[batch[i,0]]
    #     id2 = TEST_IDS[batch[i,1]]
    #     id3 = TEST_IDS[batch[i,2]]
    #     print(TEST_TRACK_INFO[id1]["name"])
    #     print(TEST_TRACK_INFO[id2]["name"])
    #     print(TEST_TRACK_INFO[id3]["name"])
    #     print()


    # trainer.train_batch(batch)