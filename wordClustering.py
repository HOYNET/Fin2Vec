import argparse
import yaml
import torch
from fin2vec import Fin2VecDataset, Config2Fin2Vec
import numpy as np
import matplotlib
from sklearn.metrics import pairwise_distances
import math

matplotlib.use("TkAgg")  # for wsl user


def replace_nan_with_nearest(tensor: torch.tensor) -> torch.tensor:
    if tensor.dim() > 1:
        for i in range(tensor.size(0)):
            tensor[i, :] = replace_nan_with_nearest(tensor[i, :])
        return tensor

    isnan = torch.isnan(tensor)
    if torch.all(isnan):
        tensor = tensor.zero_()
        return tensor

    while torch.any(isnan):
        shifted = torch.roll(isnan, 1, dims=0)
        shifted[0] = False
        tensor[isnan] = tensor[shifted]
        isnan = torch.isnan(tensor)

    return tensor


def maxPreProc(tensor: torch.tensor) -> torch.tensor:
    max = tensor.max(dim=-1, keepdim=True)[0]
    tensor /= max

    if torch.isnan(tensor).any():
        replace_nan_with_nearest(tensor)

    return tensor


parser = argparse.ArgumentParser(description="Get Path of Files.")
parser.add_argument(
    "-c",
    "--configFile",
    metavar="path",
    dest="yml",
    type=str,
    help="Path of Model Config File(.yml).",
)

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.yml) as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)
        wordClustering = yml["wordClustering"]

    device = torch.device(wordClustering["device"])

    config = wordClustering["target"]
    model, encoder, term, inputs = Config2Fin2Vec(config, device)

    data = wordClustering["data"]
    dataset = Fin2VecDataset(
        data["codeFile"],
        data["priceFile"],
        inputs,
        term,
        data["nelement"],
        period=("2020-03-01", "2022-03-01", "%Y-%m-%d"),
        cp949=True,
        ngroup=6000,
    )

    data = dataset.__getitem__(0)

    model.eval()
    word2embedding = model.word2embedding

    result = word2embedding(torch.arange(2685))
    result = result.detach().numpy()

    euclidean_distances = pairwise_distances(result, metric="euclidean")

    k = 6
    nearest_points = np.argsort(euclidean_distances, axis=1)[:, :k]

    for i, neighbors in enumerate(nearest_points):
        print(
            f"{i}th point's {k-1} nearest neighbors: {dataset.idx2code(neighbors[1:])}"
        )
        if i == 30:  
            break
