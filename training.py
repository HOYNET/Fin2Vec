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
    "-i",
    "--infoFile",
    metavar="path",
    dest="code",
    type=str,
    help="Path of Code Info File.",
)
parser.add_argument(
    "--batches",
    metavar="size",
    dest="batches",
    type=int,
    help="Size of Batch.",
)
parser.add_argument(
    "--epochs",
    metavar="size",
    dest="epochs",
    type=int,
    help="Size of Epoch.",
)
parser.add_argument(
    "--lr",
    metavar="size",
    dest="lr",
    type=float,
    help="Learning Rate.",
)
parser.add_argument(
    "-e",
    "--embeddingSize",
    metavar="size",
    dest="embeddingSize",
    type=lambda x: tuple(map(int, x.split(","))),
    help="Size of Embedding.",
)
parser.add_argument(
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
    "-s",
    "--sourceFeatures",
    metavar="size",
    dest="inputs",
    type=lambda x: x.split(","),
    help="inputFeatures.",
)
parser.add_argument(
    "-t",
    "--targetFeatures",
    metavar="size",
    dest="outputs",
    type=lambda x: x.split(","),
    help="outputFeatures.",
)
parser.add_argument(
    "-d",
    "--device",
    metavar="size",
    dest="device",
    type=str,
    help="device.('cuda:0')",
)
parser.add_argument(
    "-c",
    "--checkpointsPth",
    metavar="size",
    dest="ckpt",
    type=str,
    help="path to save checkpoints",
)

if __name__ == "__main__":
    # parse arguments
    args = parser.parse_args()
    assert (
        args.code
        and args.price
        and args.batches
        and args.epochs
        and args.lr
        and args.embeddingSize
        and args.term
        and args.inputs
        and args.outputs
    )

    # define device
    if args.device:
        device = torch.device(args.device)
    else:
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
    trainer = Trainer(model, dataset, args.batches, 0.8, optimizer, device, lossFn)

    # train
    trainer(args.epochs, args.ckpt)