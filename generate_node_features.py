import os
from os import path
import json
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")

import pandas as pd
import numpy as np
import dgl
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
    if clip_path.rsplit(".")[-1] == "wav":
        raw_clip, raw_sr = torchaudio.load(clip_path)
    else:
        raw_clip, raw_sr = librosa.load(clip_path, sr=None, mono=True)
        raw_clip = torch.tensor(raw_clip).unsqueeze(0)
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


def load_clips_from_dataset(dataset_dir):
    clips_dir = os.path.join(dataset_dir, CLIPS_SUBDIR)
    emb_dir = os.path.join(dataset_dir, "features_openl3")

    if not os.path.isdir(emb_dir):
        os.makedirs(emb_dir)

    clips = []
    
    for i,fn in enumerate(os.listdir(clips_dir)):
        name = fn.rsplit('.')[0]
        emb_pth = os.path.join(emb_dir, name + ".pt")
        if not OVERRIDE and os.path.isfile(emb_pth):
            continue
        
        clip_pth = os.path.join(clips_dir, fn)
        clip, sr = get_clip(clip_pth, SAMPLE_RATE, N_SAMPLES)

        clips.append( (clip, sr, emb_pth) )

    return clips

def download_batch_from_api(batch_ids, tracks_dict):

    temp_dir = "./temp_clips"

    for id in batch_ids:
        url = tracks_dict[id]["preview_url"]
        for i in range(0, 3):
            try:
                urllib.request.urlretrieve(url, os.path.join(temp_dir, id + ".mp3"))
                return True
            except Exception as e:
                print(e)
                print(f"An error occured, trying {id} again.")
    

def save_embeddings_to_dataset(dataset_dir, model_name):
    pass


# GENERATE CONTENT EMBEDDINGS TO REPRESENT NODES

def generate_features_openl3(dataset_dir):
    print("Generating OpenL3 embeddings...")
    
    model = openl3.models.load_audio_embedding_model(
        input_repr="mel128",
        content_type="music",
        embedding_size=512
    )

    clips_dir = os.path.join(dataset_dir, CLIPS_SUBDIR)
    emb_dir = os.path.join(dataset_dir, "features_openl3")

    if not os.path.isdir(emb_dir):
        os.makedirs(emb_dir)
    
    for i,fn in enumerate(os.listdir(clips_dir)):
        name = fn.rsplit('.')[0]
        emb_pth = os.path.join(emb_dir, name + ".pt")
        if not OVERRIDE and os.path.isfile(emb_pth):
            continue
        
        clip_pth = os.path.join(clips_dir, fn)
        clip, sr = get_clip(clip_pth, SAMPLE_RATE, N_SAMPLES)
        
        clip = clip.transpose(1,0).numpy()
        hop_size = 2 # (sec) - aka embed 1s clip every 2s and take average - decrease later!!!
        emb_batch, ts = openl3.get_audio_embedding(clip, sr, model=model, hop_size=hop_size)
        emb = torch.mean(torch.from_numpy(emb_batch), dim=0, keepdim=False)
        if(i%5 == 0):
            print(f"{i} done")

        torch.save(emb, emb_pth)

def generate_features_vggish(dataset_dir):
    print("Generating VGGish embeddings...")

    model = vgk.get_embedding_function(hop_duration=0.25)

    clips_dir = os.path.join(dataset_dir, CLIPS_SUBDIR)
    emb_dir = os.path.join(dataset_dir, "features_vggish")

    if not os.path.isdir(emb_dir):
        os.makedirs(emb_dir)

    for i,fn in enumerate(os.listdir(clips_dir)):
        name = fn.rsplit('.')[0]
        emb_pth = os.path.join(emb_dir, name + ".pt")
        if not OVERRIDE and os.path.isfile(emb_pth):
            continue
        
        clip_pth = os.path.join(clips_dir, fn)
        clip, sr = get_clip(clip_pth, SAMPLE_RATE, N_SAMPLES)

        Z, ts = model(clip, sr)
        torch.save(torch.tensor(Z), emb_pth)

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

def embed_all_clips(dataset_dir, emb_function):
    print("Generating OpenL3 embeddings...")
    
    model = openl3.models.load_audio_embedding_model(
        input_repr="mel128",
        content_type="music",
        embedding_size=512
    )

    clips_dir = os.path.join(dataset_dir, CLIPS_SUBDIR)
    emb_dir = os.path.join(dataset_dir, "features_openl3")

    if not os.path.isdir(emb_dir):
        os.makedirs(emb_dir)
    
    for i,fn in enumerate(os.listdir(clips_dir)):
        name = fn.rsplit('.')[0]
        emb_pth = os.path.join(emb_dir, name + ".pt")
        if not OVERRIDE and os.path.isfile(emb_pth):
            continue
        
        clip_pth = os.path.join(clips_dir, fn)
        clip, sr = get_clip(clip_pth, SAMPLE_RATE, N_SAMPLES)
        
        clip = clip.transpose(1,0).numpy()
        hop_size = 2 # (sec) - aka embed 1s clip every 2s and take average - decrease later!!!
        emb_batch, ts = openl3.get_audio_embedding(clip, sr, model=model, hop_size=hop_size)
        emb = torch.mean(torch.from_numpy(emb_batch), dim=0, keepdim=False)
        if(i%5 == 0):
            print(f"{i} done")

        torch.save(emb, emb_pth)


if __name__ == "__main__":

    generate_features_openl3("dataset_micro")
    #generate_features_mfcc("dataset_mini")
    #generate_features_vggish("dataset_micro")