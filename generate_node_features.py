import os
from os import path
import json
import time
import shutil

import pandas as pd
import numpy as np
import dgl
from six import b
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import torchopenl3 as openl3
import urllib
#import musicnn
#import musicnn.extractor

OVERRIDE = False
CLIPS_SUBDIR = "clips"
TEMP_CLIPS_SUBDIR = "temp_clips"
CLIP_SUFFIX = ".mp3"
EMB_PREFIX = "features_"
BATCH_SIZE = 512
SAMPLE_RATE = 16000
N_SAMPLES = 480000
DB_SCALE = True
MINMAX_NORM = True

# UTILS (borrowed from content embeddings project)

SPECTROGRAM = transform = torchaudio.transforms.MelSpectrogram(
            n_fft = 1024,
            hop_length = 512,
            n_mels = 64,
            normalized = False,
        )

def load_clip(clip_path):
    #clip_path = os.path.join(clip_dir, name + "." + suffix)
    #t1 = time.time()
    if clip_path.rsplit(".")[-1] == "wav":
        raw_clip, raw_sr = torchaudio.load(clip_path)
    else:
        raw_clip, raw_sr = librosa.load(clip_path, sr=None, mono=True)
        raw_clip = torch.tensor(raw_clip).unsqueeze(0)
    #print(time.time() - t1, "s elapsed")
    return raw_clip, raw_sr

def preprocess_clip(signal, sr, target_sr, target_samples):
    
    # resample if necessary
    if sr and sr != target_sr:
        signal = T.Resample(sr, target_sr)(signal)

    # to mono
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    # cut if necessary
    if target_samples and signal.shape[1] > target_samples:
        signal = signal[:, 0:target_samples]

    # right pad if necessary
    if target_samples and signal.shape[1] < target_samples:
        missing = target_samples - signal.shape[1]
        signal = torch.nn.functional.pad(signal, (0, missing))

    out_sr = target_sr if target_sr else sr
    return signal, out_sr

def get_clip(clip_path, target_sr, target_samples):

    raw_clip, raw_sr = load_clip(clip_path)
    clip, sr = preprocess_clip(raw_clip, raw_sr, target_sr, target_samples)
    return clip, sr

def to_spectrogram(clip):
    spec = SPECTROGRAM(clip)
    if DB_SCALE:
        spec = torchaudio.transforms.AmplitudeToDB()(spec)
    if MINMAX_NORM:
        spec -= torch.min(spec)
        spec /= torch.max(spec)
    return spec

def download_batch_from_api(batch_ids, tracks_dict, temp_dir):

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    for id in batch_ids:
        url = tracks_dict[id]["preview_url"]
        for i in range(0, 3):
            try:
                urllib.request.urlretrieve(url, os.path.join(temp_dir, id + CLIP_SUFFIX))
            except Exception as e:
                print(e)
                print(f"An error occured at id {id}. Trying once more...")
                try:
                    urllib.request.urlretrieve(url, os.path.join(temp_dir, id + CLIP_SUFFIX))
                except Exception as e:
                    print(e)
    
    print("Downloaded batch of clips.")

                

def load_batch_clips(batch_ids, clips_dir):
    
    clips = []
    for i,id in enumerate(batch_ids):
        clip_pth = os.path.join(clips_dir, id + CLIP_SUFFIX)
        clip, sr = get_clip(clip_pth, SAMPLE_RATE, N_SAMPLES)
        clips.append((clip, sr))
    return clips

def save_batch_emb(batch_ids, save_dir, embeddings):
    # save dir should be of form dataset/features_modelname

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i,id in enumerate(batch_ids):
        save_pth = os.path.join(save_dir, batch_ids[i] + ".pt")
        torch.save(embeddings[i,:].clone().detach(), save_pth)

def keep_new_ids(batch_ids, emb_dir):
    if not os.path.isdir(emb_dir):
        return batch_ids
    existing_ids = [fn.rsplit('.')[0] for fn in list(os.listdir(emb_dir))]
    new_ids = list( set(batch_ids) - set(existing_ids) )
    return new_ids

def generate_features(dataset_dir, models, online=False, load_clips=True, selection=None):
    # load clips from the dataset/clips folder generate features and save

    clips_dir = os.path.join(dataset_dir, CLIPS_SUBDIR)

    with open(os.path.join(dataset_dir, "tracks.json"), "r", encoding="utf-8") as f:
        track_dict = json.load(f)
        all_ids = list(track_dict)

    if selection is not None:
        all_ids = list(np.array(list(track_dict))[selection])
    n = len(all_ids)

    print(f"\n\033[0;33mGenerating features for {n} clips. Online = {online}.")
    print(f"Dataset: {dataset_dir}")
    print(f"Models: {list(models.keys())}\033[0m\n")

    for i in range(0, n, BATCH_SIZE):
        batch_ids = all_ids[i:min(n, i+BATCH_SIZE)]
        print(f"Batch {i}-{min(n, i+BATCH_SIZE)}")

        t_0 = time.time()

        # store yet-uncomputed ids for each model
        new_ids = {}
        for m_name in models:
            emb_dir = os.path.join(dataset_dir, EMB_PREFIX + m_name)
            new_ids[m_name] = keep_new_ids(batch_ids, emb_dir)
        
        # download needed clips into temp folder (if streamed)
        if online:
            clips_dir = os.path.join(dataset_dir, TEMP_CLIPS_SUBDIR)
            if not OVERRIDE:
                all_new_ids = set()
                for nids in new_ids.values():
                    all_new_ids = all_new_ids | set(nids)

            download_batch_from_api(list(all_new_ids), track_dict, clips_dir)

        t_dl = time.time()    

        # compute embeddings in this batch with every given model
        for m_name in models:
            print(f"Using {m_name} model:")
            emb_dir = os.path.join(dataset_dir, EMB_PREFIX + m_name)
            if not OVERRIDE:
                batch_ids = new_ids[m_name]
                print(f"{len(batch_ids)}/{BATCH_SIZE} ids are new.")
                if len(batch_ids) == 0:
                    continue
            
            paths = [os.path.join(clips_dir, cid + CLIP_SUFFIX) for cid in batch_ids]
            print("Loading batch into memory...")
            clips = load_batch_clips(batch_ids, clips_dir) if load_clips else None
            t_mem = time.time()

            embeddings = models[m_name].embed(clips, paths)
            t_emb = time.time()

            save_batch_emb(batch_ids, emb_dir, embeddings)
            t_save = time.time()

            print(f"Batch done.\nElapsed:")
            print(f"downloading clips: {t_dl-t_0}")
            print(f"loading into memory: {t_mem-t_dl}")
            print(f"embedding: {t_emb-t_mem}")
            print(f"saving embeddings: {t_save-t_emb}")



# GENERATE CONTENT EMBEDDINGS TO REPRESENT NODES

class OpenL3():

    def __init__(self):
        self.model = openl3.models.load_audio_embedding_model(
            input_repr="mel128",
            content_type="music",
            embedding_size=512
        )

    def embed(self, clips, paths):
        emb_list = []
        for clip, sr in clips:
            #clip = clip.transpose(1,0).numpy()
            hop_size = 2 # (sec) - aka embed 1s clip every 2s and take average - decrease later!!!
            batch = clip.transpose(1,0)
            emb_batch, ts = openl3.get_audio_embedding(clip, sr, model=self.model, hop_size=hop_size)
            #emb = torch.mean(torch.from_numpy(emb_batch), dim=0, keepdim=False)
            emb = torch.mean(emb_batch, dim=0, keepdim=False)
            emb_list.append(emb)

        return torch.stack(emb_list, dim=0)


# class Vggish2():

#     def __init__(self, model="MTT_vgg", layer="pool5"):
#         self.model = model
#         self.layer = layer

#     def embed(self, clips, paths):
#         emb_list = []
#         for clip_path in paths:
#             t = time.time()
#             taggram, tags, features = musicnn.extractor.extractor(clip_path,
#                                                     model=self.model,
#                                                     input_length=3,
#                                                     input_overlap=None,
#                                                     extract_features=True)
#             emb = features[self.layer]
#             emb = torch.from_numpy(emb).mean(dim=0)
#             emb_list.append(emb)
#         return torch.stack(emb_list, dim=0)


# class MusicNN():

#     def __init__(self):
#         pass

#     def embed(self, clips, paths):
#         emb_list = []
#         for clip_path in paths:
#             t = time.time()
#             taggram, tags, features = musicnn.extractor.extractor(clip_path,
#                                                     model='MTT_musicnn',
#                                                     input_length=3,
#                                                     input_overlap=None,
#                                                     extract_features=True)
#             emb = features["max_pool"]
#             # "penultimate"
#             emb = torch.from_numpy(emb).mean(dim=0)
#             emb_list.append(emb)
#         return torch.stack(emb_list, dim=0)



class RandomFeatures():

    def __init__(self, dim=512):
        self.dim = dim

    def embed(self, clips, paths):
        n = len(paths)
        return torch.rand((n, self.dim))


def generate_features_mfcc(dataset_dir):
    print("Generating MFCC embeddings...")

    mfcc = torchaudio.transforms.MFCC(SAMPLE_RATE,
        n_mfcc=40, 
        dct_type=2, 
        norm="ortho"
    )

    clips_dir = os.path.join(dataset_dir, CLIPS_SUBDIR)
    emb_dir = os.path.join(dataset_dir, "features_mfcc")

    if not os.path.isdir(emb_dir):
        os.makedirs(emb_dir)

    for i,fn in enumerate(os.listdir(clips_dir)):
        name = fn.rsplit('.')[0]
        emb_pth = os.path.join(emb_dir, name + ".pt")
        if not OVERRIDE and os.path.isfile(emb_pth):
            continue
        
        clip_pth = os.path.join(clips_dir, fn)
        clip, sr = get_clip(clip_pth, SAMPLE_RATE, N_SAMPLES)
        spec = to_spectrogram(clip)

        emb = mfcc(spec).reshape(-1, 1, 1).squeeze()
        if(i%10 == 0):
            print(f"{i} done")
            print(emb)
        torch.save(emb, emb_pth)





if __name__ == "__main__":

    models = {
        #"openl3": OpenL3(),
        #"musicnn": MusicNN(),
        #"vggish": Vggish(),
        #"vggish2": Vggish2(),
        #"vggish": Vggish2(model="MSD_vgg"),
        "random": RandomFeatures(dim=512)
    }

    with open("dataset_micro/tracks.json", "r", encoding="utf-8") as f:
        track_dict = json.load(f)
        n = len(list(track_dict))
    generate_features("dataset_micro", models, online=False, load_clips=False)


