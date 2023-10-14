import argparse
import yaml
import torch
from fin2vec import Fin2VecDataset, Config2Fin2Vec
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

matplotlib.use("TkAgg")

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
        train = yml["inference"]

    device = torch.device(train["device"])

    config = train["target"]
    model, encoder, term, inputs = Config2Fin2Vec(config, device)

    data = train["data"]
    dataset = Fin2VecDataset(
        data["codeFile"],
        data["priceFile"],
        inputs,
        term,
        period=("2020-03-01", "2022-03-01", "%Y-%m-%d"),
        cp949=True,
        ngroup=6000,
    )

    model.eval()
    word2embedding = model.word2embedding

    result = word2embedding(torch.arange(2685))

    # Torch 텐서를 Numpy 배열로 변환
    result = result.detach().numpy()

    # PCA를 사용하여 100차원 데이터를 2차원으로 축소
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(result)

    # 첫 번째 주성분과 두 번째 주성분을 시각화
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    plt.title("PCA Result")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig("./embeddings.png")
    plt.close()

    maxindex = np.argmax(result, axis=1)
    countsmaxindex = np.bincount(maxindex, minlength=100)
    index_groups = {}
    for i, max_index in enumerate(maxindex):
        if max_index in index_groups:
            index_groups[max_index].append(dataset.idx2code(i))
        else:
            index_groups[max_index] = [dataset.idx2code(i)]
    plt.figure(figsize=(15, 9))
    plt.bar(range(0, 100), countsmaxindex)
    plt.title("Max Indices of Data1")
    plt.xlabel("Row")
    plt.ylabel("Max Index")
    plt.savefig("./maxIndex.png")
    plt.close()

    print(1)
