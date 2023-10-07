import argparse
import torch
from torch import nn
from pcrn import PCRN, PCRNDataset, Config2PCRN
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

    if "pcrn" in models:
        pcrn = models["pcrn"]
        data, train, config = pcrn["data"], pcrn["train"], pcrn["config"]

        # import dataset
        dataset = PCRNDataset(
            data["codeFile"],
            data["priceFile"],
            config["inputs"],
            config["outputs"],
            cp949=True,
            term=config["term"],
        )

        # load model
        model: PCRN = Config2PCRN(config)
        if "model" in pcrn:
            model.load_state_dict(torch.load(pcrn["model"], map_location=device))
        model.to(device)

        # make trainer
        lossFn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train["lr"])
        trainer = PCRNTrainer(
            model, dataset, train["batches"], train["eval"], optimizer, device, lossFn
        )

        # train
        trainer(train["epochs"], train["ckpt"])
