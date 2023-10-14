import argparse
import yaml
import torch
from pcrn import Config2PCRN, PCRNTrainer, PCRNDataset
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
    model, term, inputs, outputs = Config2PCRN(config, device)

    data = train["data"]
    dataset = PCRNDataset(
        data["codeFile"],
        data["priceFile"],
        inputs,
        outputs,
        True,
        term,
    )

    lossFn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train["lr"])
    trainer = PCRNTrainer(
        model, dataset, train["batches"], train["trainRate"], optimizer, device, lossFn
    )

    trainer(train["epochs"], train["ckpt"])
