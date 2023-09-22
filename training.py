import argparse
from visualize import visualize
import torch
from torch import nn
from torch.utils.data import DataLoader
from parse import stockDataset
from model import Hoynet

device = torch.device("cpu")


def replace_nan_with_nearest(tensor: torch.tensor) -> torch.tensor:
    if tensor.dim() > 1:
        for i in range(tensor.size(1)):
            replace_nan_with_nearest(tensor[:, i])
        return tensor

    isnan = torch.isnan(tensor)
    while torch.any(isnan):
        # nan 값 앞의 값으로 대체
        shifted = torch.roll(isnan, 1, dims=0)
        shifted[0] = False
        tensor[isnan] = tensor[shifted]
        isnan = torch.isnan(tensor)

    return tensor


def train(dataloader, model, loss_fn, optimizer, epochs: int) -> None:
    size = len(dataloader.dataset)
    for i, batch in enumerate(dataloader):
        x, y = batch["data"], batch["label"]
        x, y = x.to(device).to(dtype=torch.float32), y.to(device).to(
            dtype=torch.float32
        )

        if torch.any(x.isnan()):
            replace_nan_with_nearest(x)
        if torch.any(y.isnan()):
            replace_nan_with_nearest(y)

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
            y, pred = y * yMax, pred * yMax
            visualize(
                y[0].detach().numpy(),
                # y[0].cpu().detach().numpy(),
                pred[0].detach().numpy(),
                # pred[0].cpu().detach().numpy(),
                epochs,
                i,
            )


def test(dataloader, model, loss_fn) -> None:
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
parser.add_argument(
    "-t",
    "--term",
    metavar="size",
    dest="term",
    type=int,
    help="Size of term.",
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
        or args.term is None
    ):
        print("Missing options ...")
        exit()

    term = args.term
    epoch = args.epochs
    # make dataLoader
    traindataset = stockDataset(args.code, args.price, True, term=args.term)
    traindataLoader = DataLoader(
        traindataset,
        batch_size=args.batchSize,
        shuffle=True,
    )

    # make model
    dummy = next(iter(traindataLoader))
    dates, inputSize, outputSize, hiddenSize, layerSize, fusionSize, embeddingSize = (
        dummy["data"].shape[-1],
        dummy["data"].shape[-2],
        dummy["label"].shape[-2],
        64,
        7,
        32,
        args.embeddingSize,
    )
    model = Hoynet(
        dates, inputSize, outputSize, hiddenSize, layerSize, fusionSize, embeddingSize
    )
    if torch.cuda.is_available():
        model.cuda()
    lossFn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training
    for t in range(0, epoch):
        print(f"Epoch {t}\n-------------------------------")
        train(traindataLoader, model, lossFn, optimizer, t)
        # test(test_dataloader, model, loss_fn)
        path = f"./checkpoints/hoynet_{t}.pth"
        torch.save(model.state_dict(), path)
