import argparse
import yaml
import torch
from fin2vec import Fin2VecDataset, Config2Fin2Vec
import math


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
        data["nelement"],
        period=("2020-03-01", "2022-03-01", "%Y-%m-%d"),
        cp949=True,
        ngroup=6000,
    )

    data = dataset.__getitem__(0)

    model.eval()
    encoder.eval()

    src, idx, msk = (
        maxPreProc(torch.tensor(data["src"], dtype=torch.float32, device=device)),
        torch.tensor(data["index"], dtype=torch.int32, device=device),
        torch.tensor(data["mask"], dtype=torch.bool, device=device),
    )
    srcShape = src.shape
    src = encoder(src) * math.sqrt(64)
    src = src.reshape(srcShape[0], -1)
    result = model(src, idx, msk)

    print(result)
