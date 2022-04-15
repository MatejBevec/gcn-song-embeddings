import os
import shutil
import torch
import pandas as pd
import numpy as np

from spotify_graph import SpotifyGraph

def export_csv(data_dir, emb_dir, knn_path=None, sample_size=None):

    ds = SpotifyGraph(data_dir, None)
    g, track_ids, _, _ = ds.to_dgl_graph()

    size = sample_size if sample_size is not None else len(track_ids)
    sample = np.random.permutation(len(track_ids))[0:size]
    id_sample = np.array(track_ids)[sample]
    titles = [ds.tracks[key]["name"] for key in id_sample]
    artists = [ds.tracks[key]["artist"] for key in id_sample]

    emb_list = []
    for id in id_sample:
        emb_pth = os.path.join(emb_dir, id + ".pt")
        emb_list.append(torch.load(emb_pth))

    emb_mat = torch.stack(emb_list, dim=0).numpy()
    print(emb_mat)
    emb_df = pd.DataFrame(data=emb_mat)
    print(emb_df)

    id_df = pd.DataFrame(data=id_sample)
    id_df["titles"] = titles
    id_df["artists"] = artists
    print(id_df)

    os.mkdir("export")
    os.mkdir("export/images")
    os.mkdir("export/clips")

    for id in id_sample:
        album_id = ds.tracks[id]["album_id"]
        img_src = os.path.join(data_dir, "images", album_id + ".jpg")
        clip_src = os.path.join(data_dir, "clips", id + ".mp3")
        try:
            shutil.copy(img_src, os.path.join("export", "images", id + ".jpg"))
        except:
            print(img_src)
            pass
        try:
            shutil.copy(clip_src, os.path.join("export", "clips"))
        except:
            pass

    emb_df.to_csv("export/emb.csv", index=None, header=None)
    id_df.to_csv("export/info.csv", index=None, header=None)

    if knn_path is not None:
        knn_w_mat, knn_n_mat = torch.load(knn_path)
        knn_w_df = pd.DataFrame(data=knn_w_mat[sample, :].numpy())
        print(knn_w_df)
        knn_n_df = pd.DataFrame(data=knn_n_mat[sample, :].numpy())
        knn_w_df.to_csv("export/knn_w.csv", index=None, header=None)
        knn_n_df.to_csv("export/knn_n.csv", index=None, header=None)


if __name__ == "__main__":
    export_csv("./dataset_small",
            "./baselines/emb/PinSageOpenL3",
            knn_path="./baselines/knn/PinSageOpenL3.pt",
            sample_size=None)
            


