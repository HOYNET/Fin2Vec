import argparse
import yaml
import torch
from fin2vec import Fin2VecDataset, Config2Fin2Vec
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

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

    euclidean_distances = pairwise_distances(result, metric="euclidean")

    # 가장 가까운 5개의 이웃을 찾기 (자기 자신을 포함하므로 k=6)
    k = 6
    nearest_points = np.argsort(euclidean_distances, axis=1)[:, :k]

    # 결과 출력
    for i, neighbors in enumerate(nearest_points):
        print(
            f"{i}th point's {k-1} nearest neighbors: {dataset.idx2code(neighbors[1:])}"
        )
        if i == 30:  # 상위 10개 데이터 포인트만 출력하여 예시로 확인합니다.
            break

    # cosine_sim = cosine_similarity(result)

    # # 임계값 설정
    # threshold = -0.4  # 예시 값

    # # 임계값 이상의 코사인 유사도를 가지는 항목 쌍 찾기
    # np.fill_diagonal(cosine_sim, -1)  # 대각선 요소를 -1로 설정하여 자기 자신과의 유사도를 제외
    # pairs = np.column_stack(np.where(cosine_sim < threshold))

    # # 결과 출력
    # for pair in pairs:
    #     print(f"0: {dataset.idx2code(pair[0])} 1:{dataset.idx2code(pair[1])}")
    # # t-SNE를 사용하여 100차원 임베딩을 2차원으로 축소
    # tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    # tsne_results = tsne.fit_transform(cosine_sim)

    # # 2D 그래프에 시각화
    # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    # plt.title("t-SNE visualization of embeddings")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # plt.show()

    # print(1)
