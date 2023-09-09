import argparse
import pandas as pd
import numpy as np
import pickle

parser = argparse.ArgumentParser(description="Get Path of Files.")
parser.add_argument(
    "-p",
    "--pickleFile",
    metavar="path",
    dest="pklpath",
    type=str,
    help="Path of Pickle File Containing Price Data.",
)


def dict2np(data: pd.DataFrame) -> np.array:
    tmp = []
    for i in data:
        tmp.append(data[i].to_numpy())
    return np.array(tmp)


if __name__ == "__main__":
    # load data
    args = parser.parse_args()
    if args.pklpath is None:
        print("Missing options ...")
        exit()
    with open(args.pklpath, "rb") as f:
        data = pickle.load(f)

    data = dict2np(data)
