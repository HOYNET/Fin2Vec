import argparse
import torch
from torch.utils.data import DataLoader
from parse import stockDataset


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    if args.code is None or args.price is None or args.batchSize is None or args.epochs:
        print("Missing options ...")
        exit()

    # make dataLoader
    traindataset = stockDataset(args.code, args.price, True)
    traindataLoader = DataLoader(traindataset, batch_size=args.batchSize)

    # make model

    # training
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # train(train_dataloader, model, loss_fn, optimizer)
        # test(test_dataloader, model, loss_fn)
