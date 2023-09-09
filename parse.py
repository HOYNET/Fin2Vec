import argparse
import pandas as pd
import numpy as np
import pickle

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
    "-d",
    "--destination",
    metavar="path",
    dest="destination",
    type=str,
    help="Destination of Result Pickle File.",
)


def parseByCode(priceData: pd.DataFrame, codeData: pd.DataFrame) -> dict:
    if priceData.empty or codeData.empty:
        return {}

    if "tck_iem_cd" not in priceData.columns or "tck_iem_cd" not in codeData.columns:
        return {}

    filtered_data_dict = {}
    priceData["tck_iem_cd"].str.strip()
    for code in codeData["tck_iem_cd"]:
        code = code.strip()
        filtered_data = priceData[priceData["tck_iem_cd"] == code].copy()
        if filtered_data.empty:
            continue
        filtered_data["trd_dt"] = pd.to_datetime(
            filtered_data["trd_dt"], format="%Y%m%d"
        )
        filtered_data.set_index("trd_dt", inplace=True)
        filtered_data.drop(columns=["tck_iem_cd"], inplace=True)
        filtered_data_dict[code] = filtered_data

    return filtered_data_dict


if __name__ == "__main__":
    args = parser.parse_args()
    if args.price is None or args.code is None or args.destination is None:
        print("Missing args ...")
        exit(-1)
    rawPrice = pd.read_csv(args.price)
    code = pd.read_csv(args.code, encoding="CP949")  # CP949 for Korean
    parsedData = parseByCode(rawPrice, code)
    destination = args.destination+"/parsedData.pkl"
    with open(destination, "wb") as pickle_file:
        pickle.dump(parsedData, pickle_file)
        print("Result is stored in "+destination)