import argparse
import torch
from torch import nn
from parse import StockDataset
from model import Hoynet
from trainer import Trainer

parser = argparse.ArgumentParser(description="Get Path of Files.")
parser.add_argument(
    "-p",
    "--priceFile",
    metavar="path",
    dest="price",
    type=str,
    help="Path of Price File.",
)
parser.add_argument(
    "-c",
    "--codeFile",
    metavar="path",
    dest="code",
    type=str,
    help="Path of Code Info File.",
)
parser.add_argument(
    "-b",
    "--batchSize",
    metavar="size",
    dest="batchSize",
    type=int,
    help="Size of Batch.",
)
parser.add_argument(
    "-e",
    "--epochSize",
    metavar="size",
    dest="epochs",
    type=int,
    help="Size of Epoch.",
)
parser.add_argument(
    "-l",
    "--learningRate",
    metavar="size",
    dest="lr",
    type=float,
    help="Learning Rate.",
)
parser.add_argument(
    "-s",
    "--embeddingSize",
    metavar="size",
    dest="embeddingSize",
    type=lambda x: tuple(map(int, x.split(","))),
    help="Size of Embedding.",
)
parser.add_argument(
    "-t",
    "--term",
    metavar="size",
    dest="term",
    type=int,
    help="Size of term.",
)
parser.add_argument(
    "-m",
    "--modelPath",
    metavar="size",
    dest="model",
    type=str,
    help="Path of Saved Model(.pth).",
)
parser.add_argument(
    "-i",
    "--inputFeatures",
    metavar="size",
    dest="inputs",
    type=lambda x: x.split(","),
    help="inputFeatures.",
)
parser.add_argument(
    "-o",
    "--outputFeatures",
    metavar="size",
    dest="outputs",
    type=lambda x: x.split(","),
    help="outputFeatures.",
)

if __name__ == "__main__":
    # parse arguments
    args = parser.parse_args()
    assert (
        args.code
        and args.price
        and args.batchSize
        and args.epochs
        and args.lr
        and args.embeddingSize
        and args.term
        and args.inputs
        and args.outputs
    )
    
    # define device
    device = torch.device("cpu")

    # import dataset
    dataset = StockDataset(
        args.code, args.price, args.inputs, args.outputs, 10, cp949=True, term=args.term
    )

    # make model
    model = Hoynet(len(args.inputs), 64, 8, 7, 0.1, len(args.outputs), device)
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # make trainer
    lossFn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, dataset, args.batchSize, 0.8, optimizer, device, lossFn)
