import numpy as np
import pandas as pd

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import urllib.request
import os, sys
from os import path
import json
import time
import random
import shutil
import pprint
import re

# if it needs Spotify API, it goes here


#TODO:
# download clips
# save as dataframe
# download images for every song
# enable getting all songs from a playlist or at least randomize offset

# dataset/tracks.csv:
# id, artists, acousticness, danceability, duration_ms, energy, explicit, instrumentalness, key, liveness, loudness, lexical

# dataset/collections.csv:
# id, type[album/playlist], num_songs, (artist), (genre), (songs[array of ids])

# dataset/graph.json
# {collections:[ids], tracks:[ids], edges[{c,t},...]}

CLIENT_ID = "c283b507348c45ca8013d3f9a75a2af5"
CLIENT_SECRET = "1801731ef5344912bda240575d411ef8"
MAX_SONGS = 50 # max number of songs used from one album (<= 50)
PLAYLIST_ALBUM_RATIO = 3 # fetch album for how many playlists
MAX_OFFSET = 1000 # max offset when searching (small means only whats popular now)
MARKET = None # none or 2 character string
DECADES = [1950,1960,1970,1980,1990,2000,2010] # query on of these decades (albums)
SMALL_IMG = False
GENRE = False

DIRECTED = False # undirected graph has additional backwards links from tracks to albums

TRACK_COLUMNS = ["id", "artists", "lexical"]
COLLECTION_COLUMNS = ["id", "type", "num_songs", "artist"]
PP = pprint.PrettyPrinter(indent=2)

class DatasetCollector():

    def __init__(self, dataset_dir, num_playlists):
        self.dir = dataset_dir # where to save data
        self.N = num_playlists # stop after N playlists

        self.sp = sp = spotipy.Spotify(\
            requests_timeout=10,\
            auth_manager=SpotifyClientCredentials(\
                client_id="c283b507348c45ca8013d3f9a75a2af5",
                client_secret="1801731ef5344912bda240575d411ef8"))

        self.tracks_pth = path.join(self.dir, "tracks.json")
        self.col_pth = path.join(self.dir, "collections.json")
        self.graph_pth = path.join(self.dir, "graph.json")

        self.clips_dir = path.join(self.dir, "clips")
        self.images_dir = path.join(self.dir, "images")

        t,c,g = self.load_or_create_dataset()
        self.tracks = t #df of tracks and info
        self.collections = c #df of albums and playlists
        self.graph = g #playlist-song graph in json form

    
    def load_or_create_dataset(self):
        
        exists = False
        if path.isfile(self.tracks_pth):
            print("Loading existing dataset.")

            with open(self.tracks_pth, "r", encoding="utf-8") as f:
                tracks = json.load(f)
            with open(self.col_pth, "r", encoding="utf-8") as f:
                collections = json.load(f)
            with open(self.graph_pth, "r", encoding="utf-8") as f:
                graph = json.load(f)

            # make backups
            for pth in [self.tracks_pth, self.col_pth, self.graph_pth]:
                shutil.copy2(pth, pth+".backup")
        else:
            print("Creating empty dataset.")

            if not os.path.isdir(self.dir):
                os.mkdir(self.dir)

            tracks = {}
            collections = {}
            graph = {"tracks": [], "collections": [], "edges": []}
        
        return tracks, collections, graph


    def save_dataset(self):
        tracks, collections, graph = self.tracks, self.collections, self.graph

        with open(self.tracks_pth, "w", encoding="utf-8") as f:
            json.dump(tracks, f, ensure_ascii=False, indent=2)  
        with open(self.col_pth, "w", encoding="utf-8") as f:
            json.dump(collections, f, ensure_ascii=False, indent=2) 
        with open(self.graph_pth, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

        for pth in [self.tracks_pth, self.col_pth, self.graph_pth]:
            try:
                os.remove(pth+".backup")
            except:
                pass

        print("Saved dataset to directory.")

    def save_dataset_as(self, new_dir):
        ndc = DatasetCollector(new_dir, 0)
        ndc.tracks, ndc.collections, ndc.graph = self.tracks, self.collections, self.graph
        ndc.save_dataset()

    def fetch_playlist(self):
        rnd_offset = random.randrange(0, MAX_OFFSET)
        query = random_query()
        results = self.sp.search(q=query, type='playlist', limit=50, offset=rnd_offset, market=MARKET)
        #print(query)
        if len(results["playlists"]["items"]) == 0:
            print("Query returned no results.")
            return
        playlist_id = results["playlists"]["items"][0]["id"]
        playlist = self.sp.playlist(playlist_id, additional_types=("track",))
        tracks = [t["track"] for t in playlist["tracks"]["items"] if t["track"]]
        track_ids = [t["track"]["id"] for t in playlist["tracks"]["items"] if t["track"]]

        print("\n Fetching playlist:\n", playlist["name"])
        # fetches and filters tracks, parses and saves everything to dataset
        self.process_tracks(track_ids, playlist, "playlist")

    def fetch_album(self):
        decades = DECADES
        decade_str = ""
        if random.random() < 0.5:
            y = decades[random.randrange(0, len(decades))]
            decade_str = f" year:{y}-{y+10}"
            print(f"\nSearching for{decade_str}.")
        query = random_query() + decade_str
        rnd_offset = random.randrange(0, MAX_OFFSET)

        results = self.sp.search(q=query, type='album', limit=50, offset=rnd_offset, market=MARKET)
        if len(results["albums"]["items"]) == 0:
            print("Query returned no results.")
            return
        album_id = results["albums"]["items"][0]["id"]
        album = self.sp.album(album_id)
        tracks = [t for t in album["tracks"]["items"] if t]
        track_ids = [t["id"] for t in album["tracks"]["items"] if t]

        print(f"\n Fetching album:\n {album['name']} ({get_year(album['release_date'])})")
        # fetches and filters tracks, parses and saves everything to dataset
        self.process_tracks(track_ids, album, "album")


    def process_tracks(self, track_ids, collection, col_type):

        collection_id = collection["id"]
        if collection_id in self.collections:
            print("Collection already in dataset.")
            return True

        #TODO: shuffle and keep max 50 tracks
        random.shuffle(track_ids)
        track_ids = track_ids[0:MAX_SONGS]

        #!!!! dont filter out known tracks, edges need to be added

        tracks_info_raw = self.sp.tracks(track_ids)["tracks"]
        # filter out songs with no preview
        tracks_info = [info for info in tracks_info_raw if info["preview_url"] != None]
        print( f"{len(tracks_info)}/{len(tracks_info_raw)} tracks have a preview." )
        track_ids = [t["id"] for t in tracks_info]

        if len(track_ids) == 0:
            print("No tracks with clips, skipping this collection.")
            return False

        
        tracks_features = self.sp.audio_features(track_ids)

        tracks_parsed = []
        new_tracks_parsed = []
        new_track_ids = [] #track ids of first seen tracks
        all_track_ids = [] #ids of confirmed tracks (in case some fail the extra check)

        for i in range(0, len(track_ids)):
            if not tracks_features[i]:
                print("Audio features unavailable, track skipped.")
                continue

            track_id, track_parsed = track_dict(tracks_info[i], tracks_features[i])
            #todo: download tracks 

            all_track_ids.append(track_id)
            self.graph["edges"].append( {"from": collection_id, "to": track_id})
            
            if not DIRECTED:
                self.graph["edges"].append( {"from": track_id, "to": collection_id})

            #print("      " + tracks_info[i]["name"])

            if track_id not in self.tracks:
                # add new track to tracks.json and graph
                new_track_ids.append(track_id)
                self.graph["tracks"].append(track_id)
                self.tracks[track_id] = track_parsed

        print(f"{len(new_track_ids)}/{len(all_track_ids)} tracks are new." )

        self.graph["collections"].append(collection_id)

        if col_type == "playlist":
            col_id, col_dict = playlist_dict(collection, all_track_ids)
        elif col_type == "album":
            col_id, col_dict = album_dict(collection, all_track_ids)
        else:
            print("wrong type!")
            raise Exception

        self.collections[col_id] = col_dict

        return True 

    def download_item(self, id, dir, ext, url):
        # download clip for one song
        for i in range(0, 3):
            try:
                urllib.request.urlretrieve(url, path.join(dir, id + ext))
                return True
            except Exception as e:
                print(e)
                print(f"An error occured, trying {id} again.")
        return False

    def download_clips(self):
        # download 30s mp3 previews for songs in dataset

        if not os.path.isdir(self.clips_dir):
                os.mkdir(self.clips_dir)

        fnames = os.listdir(self.clips_dir)
        n = len(self.tracks)
        in_folder = set() #all clips in folder
        for fname in fnames:
            in_folder.add(fname.rsplit('.')[0])

        all = set(self.tracks) #all tracks in dataset
        to_download = all - in_folder
        to_delete = in_folder - all

        print(f"{n - len(to_download)}/{n} already stored, downloading the rest.")
        print("Ctrl + C to exit.")
        if len(to_delete) > 0:
            print(f"{len(to_delete)} unexpected files present.")

        for i,track_id in enumerate(to_download):
            try:
                ret = self.download_item(track_id, self.clips_dir, ".mp3", self.tracks[track_id]["preview_url"])
            except KeyboardInterrupt:
                print("Exiting...")
                sys.exit()
            if i%50 == 0:
                print(f"{i}/{len(to_download)}")

        print("Done.")          

    def download_images(self):
        # download album covers for all songs in tracks.json

        if not os.path.isdir(self.images_dir):
                os.mkdir(self.images_dir)

        fnames = os.listdir(self.images_dir)
        in_folder = set()
        for fname in fnames:
            in_folder.add(fname.rsplit('.')[0])

        all = set([self.tracks[t]["album_id"] for t in self.tracks])
        to_download = all - in_folder
        to_delete = in_folder - all

        urls = {}
        for t in self.tracks:
            tr = self.tracks[t]
            if tr["album_id"] in to_download:
                urls[tr["album_id"]] = tr["image_url"] if not SMALL_IMG else tr["image_url_small"]

        print(f"{len(all) - len(to_download)}/{len(all)} already stored, downloading the rest.")
        print("Ctrl + C to exit.")

        for i,image_id in enumerate(urls):
            try:
                #print(urls[image_id])
                ret = self.download_item(image_id, self.images_dir, ".jpg", urls[image_id])
            except KeyboardInterrupt:
                print("Exiting...")
                sys.exit()
            if i%50 == 0:
                print(f"{i}/{len(to_download)}")
        pass

    
    def add_track_genre(self):
        # fetch info about top artist genre for every track

        tracks = [t for t in self.tracks if "genre" not in t]
        tracks_genre = []
        #tracks_info = self.sp.tracks(track_ids)["tracks"]
        n = len(tracks)

        for i in range(0, 10, 50):
            batch_ids = tracks[i: min(i+50, n)]
            tracks_info = self.sp.tracks(batch_ids)["tracks"]

            for j in range(0, len(tracks_info)):
                print(tracks_info[j]["artists"])
                genres = tracks_info[j]["artists"][0]["genres"]
                genre = genres[0] if genres else ""
                track_id = tracks[i+j]
            


    def start(self):

        for i in range(0, self.N):
            try:
                self.fetch_playlist() 
            except Exception as e:
                print(e)

            if i%PLAYLIST_ALBUM_RATIO == 0:
                try:
                    self.fetch_album() 
                except Exception as e:
                    print(e)

            if i%5 == 0:
                print(f"\n{i}/{self.N} iterations.")
                print(f"{len(self.tracks)} tracks in dataset.")

        

def random_query():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    rnd_char = chars[random.randrange(0, len(chars))]
    rnd = random.random()
    if rnd > 0.5:
        rnd_query = rnd_char + "%"
    else:
        rnd_query = "%" + rnd_char + "%"
    return rnd_query

def track_dict(info, features):
    # parse info for a single track into a dict
    track = {}
    id = info["id"]
    track["name"] = info["name"]
    track["popularity"] = info["popularity"]
    track["preview_url"] = info["preview_url"]
    track["album_id"] = info["album"]["id"]
    alb = info["album"]
    track["image_url"] = alb["images"][len(alb["images"])-2]["url"]\
        if len(alb["images"]) > 0 else None
    track["image_url_small"] = alb["images"][len(alb["images"])-1]["url"]\
        if len(alb["images"]) > 0 else None
    track["artist_id"] = alb["artists"][0]["id"]\
        if len(alb["artists"]) > 0 else None
    track["artist"] = alb["artists"][0]["name"]\
        if len(alb["artists"]) > 0 else None

    if GENRE:
        genres = alb["artists"][0]["genres"]
        track["genre"] = genres[0] if genres else ""

    track["danceability"] = features["danceability"]
    track["energy"] = features["energy"]
    track["key"] = features["key"]
    track["loudness"] = features["loudness"]
    track["mode"] = features["mode"]
    track["acousticness"] = features["acousticness"]
    track["instrumentalness"] = features["instrumentalness"]
    track["liveness"] = features["liveness"]
    track["valence"] = features["valence"]
    track["tempo"] = features["tempo"]
    return id, track

def album_dict(data, track_ids):
    # parse info for an album into a dict
    album = {}
    id = data["id"]
    album["type"] = "album"
    album["name"] = data["name"]
    album["num_tracks"] = len(track_ids)
    album["artist"] = data["artists"][0]["name"] if len(data["artists"]) > 0 else None
    album["year"] = get_year(data["release_date"])
    album["image_url"] = data["images"][len(data["images"])-2]["url"] if len(data["images"]) > 0 else None
    album["ztracks"] = track_ids
    return id, album

def playlist_dict(data, track_ids):
    # parse info for a playlist into a dict
    playlist = {}
    id = data["id"]
    playlist["type"] = "playlist"
    playlist["name"] = data["name"]
    playlist["num_tracks"] = len(track_ids)
    playlist["description"] = data["description"]
    #playlist["image_url"] = data["images"][len(data["images"])-1]["url"] if len(data["images"]) > 0 else None
    playlist["ztracks"] = track_ids
    return id, playlist

def get_year(date_str):
    return int(re.sub(r'-.*', "", date_str))


if __name__ == "__main__":

    if len(sys.argv) < 2 or (sys.argv[1] not in ["create", "clips", "images", "genre"]):
        print("Unrecognized command.")
        exit()
    mode = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    dc = DatasetCollector("dataset", n)

    #dc.add_track_genre() --> # "genres" not available with this version of API?

    try:
        if mode == "create":
            dc.start()
        elif mode == "clips":
            dc.download_clips()
        elif mode == "images":
            dc.download_images()
        elif mode == "genre":
            dc.add_track_genre()
    except KeyboardInterrupt:
        print("Exiting...")

    dc.save_dataset()
