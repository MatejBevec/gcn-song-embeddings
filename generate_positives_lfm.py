from dataclasses import dataclass
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
import math
import random

from spotify_graph import SpotifyGraph
import pinsage_model as psm

PRECOMP_NAME = "./neighborhoods_micro.pt"


def get_lfm_triplets(lfm_path):
    # return (track, artist, album) for all tracks in LFM

    le_path = os.path.join(lfm_path, "LFM-1b_LEs.txt")

    tr_path = os.path.join(lfm_path, "LFM-1b_tracks.txt")
    ar_path = os.path.join(lfm_path, "LFM-1b_artists.txt")
    al_path = os.path.join(lfm_path, "LFM-1b_albums.txt")
    tracks = pd.read_csv(tr_path, sep="\t", names=("track-id", "track-name", "artist-id"), index_col=0)
    artists = pd.read_csv(ar_path, sep="\t", names=("artist-id", "artist-name"), index_col=0)
    albums = pd.read_csv(al_path, sep="\t", names=("album-id", "album-name", "artist-id"), index_col=0)

    triplets = []
    triplets_set = set()

    size = 100000
    reader = pd.read_csv(le_path, sep="\t", chunksize=size,
            names=("user-id", "artist-id", "album-id", "track-id", "timestamp"))

    le_df = pd.DataFrame()

    for i, chunk in enumerate(reader):
        print(i)
        if i == 9000:
            break
        if random.random() < 0.01:
            #sample = chunk.sample(frac=0.01)
            le_df = pd.concat([le_df, chunk])
        

    for i, row in le_df.iterrows():
        t_id, ar_id, al_id = row["track-id"], row["artist-id"], row["album-id"]
        id_triplet = (t_id, ar_id, al_id)
        try: 
            triplet = (tracks["track-name"][t_id], artists["artist-name"][ar_id], albums["album-name"][al_id],
                        id_triplet, row["user-id"], row["timestamp"])
            triplets_set.add(id_triplet)
        except Exception as e:
            print(e)
            continue
        triplets.append(triplet)

    print("le tracks: ", len(triplets_set))

    return_df = pd.DataFrame(list(triplets), columns=("track-name", "artist-name", "album-name",
                                            "id", "user-id", "timestamp"))

    return return_df


def get_lfm_spotify_map(lfm_triplets, sp_tracks_path, match="triplets"):

    with open(sp_tracks_path, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    has_alb = [t for t in tracks if "album" in t]
    print("Total:", len(tracks))
    print("Has album:", len(has_alb))
    
    spotify_tracks = pd.DataFrame.from_dict(tracks, orient="index")[["name", "artist", "album"]]
    spotify_tracks = spotify_tracks.applymap(lambda s:s.lower() if type(s) == str else s)
    spotify_tracks["spotify-id"] = spotify_tracks.index

    lfm_triplets = lfm_triplets.applymap(lambda s:s.lower() if type(s) == str else s)

    if match == "triplets":
        df = lfm_triplets.merge(spotify_tracks, how="inner",
                        left_on=["track-name", "artist-name", "album-name"],
                        right_on=["name", "artist", "album"])
    if match == "doubles":
        df = lfm_triplets.merge(spotify_tracks, how="inner",
                        left_on=["track-name", "artist-name"],
                        right_on=["name", "artist"])

    print(df.drop_duplicates(subset=["track-name", "artist-name", "album-name", "spotify-id"]))
    df.drop_duplicates(subset=["track-name", "artist-name", "album-name", "spotify-id"]).to_csv("test.csv")
    
    map = {}
    for i, row in df.iterrows():
        #triplet = (row["track-id"], row["artist-id"], row["album-id"])
        triplet = (row["id"])
        if triplet in map:
            map[triplet].append(row["spotify-id"])
        else:
            map[triplet] = [row["spotify-id"]]

    return map





def generate_lfm_positives(le_df, lfm_spotify_map, n):

    positives = []
    triplets = lfm_spotify_map.keys()
    les = le_df[le_df["id"].isin(triplets)]

    print("filtered les: ", len(les))

    rnd_vec = np.random.permutation(len(les)-1)[0:n]
    for i in rnd_vec:
        arow = les.iloc[i, :]
        brow = les.iloc[i+1, :]
        at = datetime.fromtimestamp(arow["timestamp"])
        bt = datetime.fromtimestamp(brow["timestamp"])
        delta_t = abs((bt - at).total_seconds())

        # only consider pairs listened to within one hour
        if delta_t < 3600:
            continue
        
        if arow["user-id"] != brow["user-id"]:
            continue

        a_trip = arow["id"]
        b_trip = brow["id"]

        a_ids = lfm_spotify_map[a_trip]
        b_ids = lfm_spotify_map[b_trip]

        a_id = np.random.randint(len(a_ids))
        b_id = np.random.randint(len(b_ids))
        
        pos = {
            "a": a_ids[a_id],
            "b": b_ids[b_id]
        }

        # ignore repeated listens of the same song
        if not (pos["a"] == pos["b"]):
            positives.append(pos)

        print("positive pairs: ", len(positives))

    return positives



    

if __name__ == "__main__":

    #data_dir = sys.argv[1]
    data_dir = "dataset_small"
    n = 500000

    print(f"\033[0;33mGenerating positive training pairs for {data_dir}\
        from LFM listening events...\033[0m")

    triplets = get_lfm_triplets("./LFM")

    lfm_spotify_map = get_lfm_spotify_map(triplets, os.path.join(data_dir, "tracks.json"), match="doubles")
    print("mapped tracks: ", len(lfm_spotify_map.keys()))

    positives = generate_lfm_positives(triplets, lfm_spotify_map, n)
    for i in range(10):
        print(positives[i])

    save_path = os.path.join(data_dir, "positives_lfm_test.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(positives, f, ensure_ascii=False, indent=2)    
