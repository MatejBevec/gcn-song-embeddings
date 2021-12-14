import os
from os import path
import json
import time
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")

import pandas as pd
import numpy as np
import dgl
from six import b
from tensorflow.python.training.tracking.base import Trackable
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import openl3
import vggish_keras as vgk
import urllib

OVERRIDE = False
CLIPS_SUBDIR = "clips"
CLIP_SUFFIX = ".mp3"
EMB_PREFIX = "features_"
BATCH_SIZE = 16
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

def download_batch_from_api(batch_ids, tracks_dict):

    temp_dir = "./temp_clips"

    for id in batch_ids:
        url = tracks_dict[id]["preview_url"]
        for i in range(0, 3):
            try:
                urllib.request.urlretrieve(url, os.path.join(temp_dir, id + CLIP_SUFFIX))
                return True
            except Exception as e:
                print(e)
                print(f"An error occured, trying {id} again.")
    


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
        print(save_pth, embeddings[i, :4])

def keep_new_ids(batch_ids, emb_dir):
    if not os.path.isdir(emb_dir):
        return batch_ids
    existing_ids = [fn.rsplit('.')[0] for fn in list(os.listdir(emb_dir))]
    new_ids = list( set(batch_ids) - set(existing_ids) )
    return new_ids

def generate_features(dataset_dir, models, online=False):
    # load clips from the dataset/clips folder generate features and save

    clips_dir = os.path.join(dataset_dir, CLIPS_SUBDIR)
    # if online:
    #     with open(os.path.join(dataset_dir, "tracks.json"), "r", encoding="utf-8") as f:
    #         track_dict = json.load(f)
    #         all_ids = list(track_dict)
    # else:
    #     all_ids = [fn.rsplit('.')[0] for fn in sorted(os.listdir(clips_dir))]
    # 

    with open(os.path.join(dataset_dir, "tracks.json"), "r", encoding="utf-8") as f:
        track_dict = json.load(f)
        all_ids = list(track_dict)
    n = len(all_ids)

    print(f"Generating features for {n} clips. Online = {online}.")

    for i in range(0, n, BATCH_SIZE):
        batch_ids = all_ids[i:min(n, i+BATCH_SIZE)]
        print(f"Batch {i}-{min(n, i+BATCH_SIZE)}")

        if online:
            download_batch_from_api(batch_ids, track_dict)      

        for m_name in models:
            print(f"Using {m_name} model:")
            emb_dir = os.path.join(dataset_dir, EMB_PREFIX + m_name)
            if not OVERRIDE:
                batch_ids = keep_new_ids(batch_ids, emb_dir)
                print(f"{len(batch_ids)}/{BATCH_SIZE} ids are new.")
                if len(batch_ids) == 0:
                    continue

            clips = load_batch_clips(batch_ids, clips_dir) # better if it could be outside for loop 
            embeddings = models[m_name].embed(clips)

            save_batch_emb(batch_ids, emb_dir, embeddings)




# GENERATE CONTENT EMBEDDINGS TO REPRESENT NODES

class OpenL3():

    def __init__(self):
        self.model = openl3.models.load_audio_embedding_model(
            input_repr="mel128",
            content_type="music",
            embedding_size=512
        )

    def embed(self, clips):
        emb_list = []
        for clip, sr in clips:
            clip = clip.transpose(1,0).numpy()
            hop_size = 2 # (sec) - aka embed 1s clip every 2s and take average - decrease later!!!
            emb_batch, ts = openl3.get_audio_embedding(clip, sr, model=self.model, hop_size=hop_size)
            emb = torch.mean(torch.from_numpy(emb_batch), dim=0, keepdim=False)
            emb_list.append(emb)

        return torch.stack(emb_list, dim=0)

class Vggish():

    def __init__(self):
        self.model = vgk.get_embedding_function(hop_duration=0.25)

    def embed(self, clips):
        emb_list = []
        for clip, sr in clips:
            Z, ts = self.model(clip, sr)
            emb_list.append(torch.tensor(Z))
    
        return torch.stack(emb_list, dim=0)


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

    #generate_features_openl3("dataset_micro")
    #generate_features_mfcc("dataset_mini")
    #generate_features_vggish("dataset_micro")

    batch_ids = [
        "0dIoGTQXDh1wVnhIiSyYEa",
        "0dRY4OrSY53yUjVgfgne1W",
        "0DwVfCYLrVXgvejYbWwZAd",
        "0EdgK7ASb4kfRkW8pVMN02"
    ]

    embeddings = torch.tensor([
        [1,2,3],
        [2,2,2],
        [4,2,0],
        [6,6,6]
    ])

    models = {
        "openl3test": OpenL3()
    }

    generate_features("dataset_micro", models)
