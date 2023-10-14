import matplotlib
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import LogLocator

matplotlib.use("TkAgg")

parser = argparse.ArgumentParser(description="Get Path of Files.")
parser.add_argument(
    "-f",
    "--lossFile",
    metavar="path",
    dest="lossFile",
    type=str,
    help="File Contains Loss Info",
)

parser.add_argument(
    "-i",
    "--lossInfo",
    metavar="path",
    dest="lossInfo",
    type=str,
    help="Loss Info",
)

parser.add_argument(
    "-l",
    "--logScale",
    metavar="path",
    dest="logScale",
    type=bool,
    help="Loss Info",
)


def read_losses(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    train_losses = []
    test_losses = []
    for line in lines:
        if "Train Loss" in line:
            train_losses.append(float(line.split(":")[1].strip()))
        elif "Test  Loss" in line:
            test_losses.append(float(line.split(":")[1].strip()))

    if len(train_losses) != len(test_losses):
        length = min(len(train_losses), len(test_losses))
        train_losses = train_losses[:length]
        test_losses = test_losses[:length]
    return train_losses, test_losses


def plot_losses(info, train_losses, test_losses, logScale=False):
    epochs = list(range(1, len(train_losses) + 1))

    ylabel = "Loss"
    if logScale:
        plt.yscale("log")
        ylabel += "_logscale"
        plt.gca().yaxis.set_major_locator(LogLocator(base=10, numticks=15))
        plt.gca().yaxis.set_minor_locator(LogLocator(base=10, subs="all", numticks=15))

    plt.plot(epochs, train_losses, marker="^", label="Train Loss")
    plt.plot(epochs, test_losses, marker="s", label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Train and Test Loss Over Epochs : {info}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{info}.png")
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.lossFile and args.lossInfo

    file_path = args.lossFile

    train_losses, test_losses = read_losses(file_path)

    plot_losses(args.lossInfo, train_losses, test_losses, args.logScale)
