import argparse
from visualize import visualize
import torch
from torch import nn
from torch.utils.data import DataLoader
from parse import stockDataset
from model import Hoynet

device = torch.device("cpu")


def train(dataloader, model, loss_fn, optimizer, epochs: int):
    size = len(dataloader.dataset)
    for i, batch in enumerate(dataloader):
        x, y = batch["data"], batch["label"]
        x, y = x.to(device).to(dtype=torch.float32), y.to(device).to(
            dtype=torch.float32
        )

        # normalization
        # x = torch.layer_norm(x, normalized_shape=x.shape)
        # y = torch.layer_norm(y, normalized_shape=y.shape)
        # Max normalization
        xMax, yMax = x.max(dim=-1, keepdim=True)[0], y.max(dim=-1, keepdim=True)[0]
        x, y = x / xMax, y / yMax

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            loss, current = loss.item(), (i + 1) * len(x)
            print(f"loss: {loss}  [{current:>5d}/{size:>5d}]")
            x, pred = x * xMax, pred * yMax
            visualize(
                x[0].detach().numpy(),
                pred[0].detach().numpy(),
                epochs,
                i,
                ["gts_iem_ong_pr", "gts_iem_hi_pr", "gts_iem_low_pr"],
            )


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

if __name__ == "__main__":
    # parse arguments
    args = parser.parse_args()
    if (
        args.code is None
        or args.price is None
        or args.batchSize is None
        or args.epochs is None
        or args.lr is None
        or args.embeddingSize is None
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
        args.embeddingSize,
    )
    model = Hoynet(dates, inputSize, hiddenSize, layerSize, fusionSize, embeddingSize)
    lossFn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # training
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(traindataLoader, model, lossFn, optimizer, t)
        # test(test_dataloader, model, loss_fn)
        if t % 5 == 0:
            path = "./checkpoints/hoynet_" + str(t) + ".pth"
            torch.save(model.state_dict(), path)
