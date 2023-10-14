import argparse
import yaml
import torch
from data2vec import Cofing2Data2Vec
from fin2vec import Fin2VecDataset, Fin2VecTrainer, Config2Fin2Vec
from torch import nn

parser = argparse.ArgumentParser(description="Get Path of Files.")
parser.add_argument(
    "-c",
    "--configFile",
    metavar="path",
    dest="yml",
    type=str,
    help="Path of Model Config File(.yaml).",
)

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.yml) as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)
        train = yml["train"]

    device = torch.device(train["device"])

    config = train["target"]
    model, encoder, term, inputs = Config2Fin2Vec(config, device)

    config = train["data2vec"]
    data2vec = Cofing2Data2Vec(config, model, encoder, device)

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

    lossFn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train["lr"])
    trainer = Fin2VecTrainer(
        data2vec,
        model,
        dataset,
        train["batches"],
        train["testRate"],
        optimizer,
        device,
        lossFn,
    )

    trainer(train["epochs"], train["ckpt"])
