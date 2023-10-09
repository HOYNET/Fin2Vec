import argparse
import torch
from torch import nn
from pcrn import Config2PCRN
from fin2vec import Fin2Vec, Fin2VecDataset, Fin2VecTrainer
from Data2vec import Data2vec
import yaml

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
        yml = yml["HoynetConfig"]

    # define device
    if "device" in yml:
        device = torch.device(yml["device"])
    else:
        device = torch.device("cpu")

    models = yml["models"]

    if "fin2vec" in models:
        fin2vec = models["fin2vec"]
        pcrn = fin2vec["pcrn"]
        data, config, train = pcrn["data"], pcrn["config"], fin2vec["train"]

        

        # load encoder
        encoder= Config2PCRN(config)
        if "encoder" in fin2vec:
            encoder.load_state_dict(torch.load(fin2vec["encoder"], map_location=device))

        # import dataset
        dataset = Fin2VecDataset(
            data["codeFile"],
            data["priceFile"],
            config["inputs"],
            term=config["term"],
            period=("2020-03-01", "2022-03-01", "%Y-%m-%d"),
            cp949=True,
        )

        # load fin2vec
        config = fin2vec["config"]
        model= Fin2Vec(
            encoder,
            dataset.__len__(),
            tuple(tuple(map(int, config["embeddings"].split(",")))),
            config["outputs"],
            config["d_model"],
            config["nhead"],
            config["d_hid"],
            config["nlayers"],
            config["dropout"],
        )
        if "model" in fin2vec:
            model.load_state_dict(torch.load(fin2vec["model"], map_location=device))


        # load data2vec
        d2v= Data2vec(
            model,
            fin2vec,
        )
        model.to(device)

        # make trainer
        lossFn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train["lr"])
        trainer = Fin2VecTrainer(
            model, d2v, dataset, train["batches"], train["eval"], optimizer, device, fin2vec, lossFn
        )

        # train
        trainer(train["epochs"], train["ckpt"])
