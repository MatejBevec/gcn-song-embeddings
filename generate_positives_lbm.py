import pandas as pd
import numpy as np
import dgl
import torch
import json
import h5py
import os
from scipy import sparse

from spotify_graph import SpotifyGraph
import pinsage_model as psm

PRECOMP_NAME = "./neighborhoods_micro.pt"

def save_dataset_intersect(lbm_map_path, tracks_path, save_path):

    maps = pd.read_csv(lbm_map_path, sep="\t")
    unique_ids = np.unique(maps["URI"])
    print(len(unique_ids))
    print(len(maps))
    with open(tracks_path, "r", encoding="utf-8") as f:
        tracks = list(json.load(f))
    print(len(tracks))

    shared_maps = maps[maps["URI"].isin(tracks)]
    print(len(shared_maps))

    for id in tracks:
        found = maps[maps["URI"] == id]
        if len(found) > 1:
            print(found)

    shared_maps.to_csv(save_path, index=False)


def read_lfm_LEs(lfm_path):
    with open(lfm_path, "r", encoding="utf-8") as f:
        line = f.readline()
        print(line)

def read_lfm_tracks(lfm_path):
    # Returns (tr_index, title, artist)
    # with open(lfm_path, "r", encoding="utf-8") as f:
    #     line = f.readline()
    #     print(line)
    tr_path = os.path.join(lfm_path, "LFM-1b_tracks.txt")
    ar_path = os.path.join(lfm_path, "LFM-1b_artists.txt")
    al_path = os.path.join(lfm_path, "LFM-1b_albums.txt")
    tracks = pd.read_csv(tr_path, sep="\t", names=("track-id", "track-name", "artist-id"))
    artists = pd.read_csv(ar_path, sep="\t", names=("artist-id", "artist-name"), index_col=0)
    artists = pd.read_csv(al_path, sep="\t", names=("album-id", "album-name", "artist-id"), index_col=0)

    print(tracks)
    print(artists)

    df = tracks.join(artists, on="artist-id")
    # TRACKS DONT HAVE AN ALBUM ID?????? wtf
    print(df[["track-id", "track-name", "artist-name"]])
    return df[["track-id", "track-name", "artist-name"]]

def query_spotify_ids(lfm_tracks, sp_tracks_path):
    with open(sp_tracks_path, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    print(lfm_tracks[["track-name", "artist-name"]])

    #tracks_artists = [f"{tr['title']} {tr['artist']}".lower() for tr in tracks]
    #spotify_ids = list(tracks.keys())

    spotify_tracks = pd.DataFrame.from_dict(tracks, orient="index")[["name", "artist"]]
    spotify_tracks = spotify_tracks#.applymap(lambda s:s.lower())
    print(spotify_tracks)

    lfm_tracks[["track-name", "artist-name"]] = lfm_tracks[["track-name", "artist-name"]]#\
                                                    #.applymap(lambda s:s.lower())

    df = lfm_tracks.merge(spotify_tracks, how="inner",
                        left_on=["track-name", "artist-name"],
                        right_on=["name", "artist"])
    print(df.to_string())


def query_spotify_with_names(lfm_path, tracks_path, save_path):
    
    
    pass

    

if __name__ == "__main__":
    
    # save_dataset_intersect("../LFM-1b/LFM-1b_spotify_URIs.tsv",
    #                         "./dataset_micro/tracks.json",
    #                         "../LFM-1b/lfm_to_spotify_map.csv")

    #read_lfm_LEs("../LFM-1b/LFM-1b_LEs.txt")
    lfm_tracks = read_lfm_tracks("../LFM-1b")
    query_spotify_ids(lfm_tracks, "./dataset_micro/tracks.json")