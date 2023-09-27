import argparse
from visualize import visualize
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from parse import stockDataset
from model import Hoynet

device = torch.device("cuda")


def replace_nan_with_nearest(tensor: torch.tensor) -> torch.tensor:
    if tensor.dim() > 1:
        for i in range(tensor.size(0)):
            tensor[i, :] = replace_nan_with_nearest(tensor[i, :])
        return tensor

    isnan = torch.isnan(tensor)
    if torch.all(isnan):
        tensor = tensor.zero_()
        return tensor

    while torch.any(isnan):
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

        xMax, yMax = x.max(dim=-1, keepdim=True)[0], y.max(dim=-1, keepdim=True)[0]
        x, y = x / xMax, y / yMax

        if torch.isnan(x).any():
            replace_nan_with_nearest(x)
        if torch.isnan(y).any():
            replace_nan_with_nearest(y)

        # forward
        pred = model(x)
        assert not torch.isnan(x).any()
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            loss, current = loss.item(), (i + 1) * len(x)
            print(f"loss: {loss}  [{current:>5d}/{size:>5d}]")
            y, pred = y * yMax, pred * yMax
            visualize(
                y[0].cpu().detach().numpy(),
                pred[0].cpu().detach().numpy(),
                epochs,
                i,
            )


def test(dataloader, model, loss_fn) -> None:
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y = batch["data"], batch["label"]
            x, y = x.to(device).to(dtype=torch.float32), y.to(device).to(
                dtype=torch.float32
            )
            xMax, yMax = x.max(dim=-1, keepdim=True)[0], y.max(dim=-1, keepdim=True)[0]
            x, y = x / xMax, y / yMax
            if torch.isnan(x).any():
                replace_nan_with_nearest(x)
            if torch.isnan(y).any():
                replace_nan_with_nearest(y)

            # forward
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
    model.train()
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
parser.add_argument(
    "-m",
    "--modelPath",
    metavar="size",
    dest="model",
    type=str,
    help="Path of Saved Model(.pth).",
)
parser.add_argument(
    "-n",
    "--numWorkers",
    metavar="size",
    dest="numWorkers",
    type=int,
    help="numWorkers.",
)
parser.add_argument(
    "-d",
    "--device",
    metavar="size",
    dest="device",
    type=str,
    help="device.('cuda:0')",
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
    )

    term = args.term
    epoch = args.epochs
    # make dataLoader
    dataset = stockDataset(args.code, args.price, True, term=args.term)
    trainLength = int(len(dataset) * 0.8)
    traindataset, testdataset = random_split(
        dataset, [trainLength, len(dataset) - trainLength]
    )
    traindataLoader = DataLoader(
        traindataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.numWorkers,
        pin_memory=True,
    )
    testdataLoader = DataLoader(testdataset, batch_size=args.batchSize, shuffle=True)

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
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location="cpu"))

    assert args.device
    device = torch.device(args.device)
    assert torch.cuda.is_available()
    model.to(device)

    lossFn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training
    for t in range(0, epoch):
        print(f"Epoch {t}\n-------------------------------")
        train(traindataLoader, model, lossFn, optimizer, t)
        test(testdataLoader, model, lossFn)
        # test(test_dataloader, model, loss_fn)
        path = f"./checkpoints/hoynet_{t}.pth"
        torch.save(model.state_dict(), path)
