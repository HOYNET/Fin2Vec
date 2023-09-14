import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from parse import stockDataset
from model import Hoynet

device = torch.device("cpu")


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for i, batch in enumerate(dataloader):
        x, y = batch["data"], batch["label"]
        x, y = x.to(device).to(dtype=torch.float32), y.to(device).to(
            dtype=torch.float32
        )

        # normalization
        x = torch.layer_norm(x, normalized_shape=x.shape)
        y = torch.layer_norm(y, normalized_shape=y.shape)

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            loss, current = loss.item(), (i + 1) * len(x)
            print(f"loss: {loss}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


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

if __name__ == "__main__":
    # parse arguments
    args = parser.parse_args()
    if (
        args.code is None
        or args.price is None
        or args.batchSize is None
        or args.epochs is None
    ):
        print("Missing options ...")
        exit()

    # make dataLoader
    traindataset = stockDataset(args.code, args.price, True)
    traindataLoader = DataLoader(
        traindataset,
        batch_size=args.batchSize,
        shuffle=True,
    )

    # make model
    dummy = next(iter(traindataLoader))["data"]
    dates, inputSize, hiddenSize, layerSize, fusionSize, embeddingSize = (
        dummy.shape[-1],
        dummy.shape[-2],
        64,
        7,
        32,
        (5, 5),
    )
    model = Hoynet(dates, inputSize, hiddenSize, layerSize, fusionSize, embeddingSize)
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    # training
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(traindataLoader, model, lossFn, optimizer)
        # test(test_dataloader, model, loss_fn)
