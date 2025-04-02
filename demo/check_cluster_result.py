from collections import defaultdict
import pickle
import torch

clusters = pickle.load(open("data/trajectoryAfterCluster.pickle", "rb"))
hash2cluster = pickle.load(open("data/sampleHash2Cluster.pickle", "rb"))


def load_data():
    file_path = "./data/trajs.pt"
    data = torch.load(file_path)
    root = []
    paths = []
    for sample in data:
        tra = sample["root"][:60, :]
        tra = tra - tra[0, :]
        root.append(tra)
        paths.append(sample["hash"])
    print("trajectorres number: ", len(root))
    return root, paths


trajs, hashes = load_data()

hash2traj = dict(zip(hashes, trajs))

cluster2candidates = defaultdict(list)
cluster2represent = {}
for hash, cluster_label in hash2cluster.items():
    traj = hash2traj[hash]
    cluster2candidates[cluster_label].append(traj)
    cluster2represent[cluster_label] = clusters[cluster_label]


# plot
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

os.makedirs("data/imgs", exist_ok=True)
for cluster_label, trajs in tqdm(cluster2candidates.items()):
    represent = cluster2represent[cluster_label]
    for traj in trajs:
        plt.plot(traj[:, 0], traj[:, 1], color="blue")
    plt.plot(represent[:, 0], represent[:, 1], color="red")
    plt.savefig(f"data/imgs/cluster_{cluster_label}.jpg")
    plt.clf()
