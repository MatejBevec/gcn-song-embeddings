import os
import json
import random
import glob
import shutil

from spotify_graph import SpotifyGraph
from pinsage_training import PinSage, train_and_save
import pinsage_training as pt
import generate_node_features as gnf
import generate_positives as gp
from baselines import EmbLoader
import eval

RUNS_DIR = "./runs_gs"

def train_run(dataset, train_positives, param_set, run_id):
    # Train and eval PinSave with a particular hyperparameter set

    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    run_name = f"gridsearch#{run_id}"
    trainer = PinSage(g, len(track_ids), features, train_positives, log=False, load_save=False)    

    # Set hyperparameters: BAD PRACTICE BUT ITS GONNA WORK
    for param in param_set:
        exec(f"trainer.{param} = {param_set[param]}")
    trainer.run_name = run_name

    # Train and embed
    trainer.train() 
    pt.save_embeddings(trainer, dataset, base_run_dir=RUNS_DIR)

def eval_run(dataset, train_positives, test_positives, run_id):

    g, track_ids, col_ids, features = dataset.to_dgl_graph()
    run_name = f"gridsearch#{run_id}"

    # Evaluate
    model = {run_name: EmbLoader(os.path.join(RUNS_DIR, run_name, "emb"))}
    bl_dir = "./baselines_lfm"
    knn_dict = eval.get_knn_dict(model, g, track_ids, train_positives, test_positives, features, bl_dir)
    _, knn_mat = knn_dict[run_name]

    mrr_score = eval.mrr(knn_mat, test_positives, knn_mat.shape[1], 1)
    hitrate_score = eval.hit_rate(knn_mat, test_positives, 100)

    return mrr_score, hitrate_score

def _all_configs(lengths, configs, d, my_config):
    if d == len(lengths):
        list_config = [int(token) for token in my_config.rsplit(".")[1:]]
        configs.append(list_config)
        return
    test = my_config
    for i in range(lengths[d]):
        new_config = f"{my_config}.{i}"
        _all_configs(lengths, configs, d+1, new_config)

def all_configs(lengths):
    configs = []
    _all_configs(lengths, configs, 0, "")
    return configs

def get_param_sets(param_grid):
    lenghts = [len(v) for v in param_grid.values()]
    params = list(param_grid.keys())
    configs = all_configs(lenghts)

    param_sets = {}
    for config in configs:
        run_id = ".".join([str(choice) for choice in config])
        param_set = {}
        for i in range(len(config)):
            param_set[params[i]] = param_grid[params[i]][config[i]]

        param_sets[run_id] = param_set

    return param_sets

def grid_search(dataset, train_positives, test_positives, param_grid, results_path="grid_search.json"):
    param_sets = get_param_sets(param_grid)
    results = {}

    # for pth in glob.glob(f"{RUNS_DIR}/gridsearch*"):
    #     shutil.rmtree(pth)

    for run_id in param_sets:
       param_set = param_sets[run_id]
       train_run(dataset, train_positives, param_set, run_id)

    for run_id in param_sets:
        param_set = param_sets[run_id]
        mrr, hitrate = eval_run(dataset, train_positives, test_positives, run_id)
        results[run_id] = {
            "params": param_set,
            "mrr": mrr,
            "hitrate@100": hitrate
        }

    results_sorted = dict(sorted(results.items(), key=lambda item: item[1]["mrr"], reverse=True))

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results_sorted, indent=4, sort_keys=False))

    return results_sorted
        



if __name__ == "__main__":

    # param_grid = {
    #     "T": [3, 10], # top 3 have T=3, otherwise poor correlation
    #     "lr": [1e-3, 5e-5], # poor correlation
    #     "epochs": [10, 30], # top 8/9 have epochs=30, so almost perfect correlation
    #     "n_layers": [2, 4], # top 4/5 have n_layers=4, otherwise poor correlation
    #     "hard_negatives": [False]
    # }

    # GRID SEARCH 3: USING TRAIN/TEST SPLIT

    param_grid = {
        "T": [3],
        "lr": [1e-4],
        "epochs": [3, 30],
        "n_layers": [2, 4],
        "hard_negatives": [False]
    }

    dataset = SpotifyGraph("./dataset_small", "./dataset_small/features_openl3")
    train_positives, test_positives = dataset.load_positives_split("./dataset_small/positives_lfm_large.json")

    results = grid_search(dataset, train_positives, test_positives, param_grid)

    for r in results:
        print(r)
        print(results[r])

    # mrr, hit_rate = eval_run(dataset, positives, "0.0.0.0.0")
    # print(mrr)
    # print(hit_rate)


