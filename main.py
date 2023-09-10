import argparse
from torch.utils.data import DataLoader
from parse import stockDataset

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

if __name__ == "__main__":
    # load data
    args = parser.parse_args()
    if args.code is None or args.price is None:
        print("Missing options ...")
        exit()
    dataLoader = DataLoader(stockDataset(args.code, args.price, True))