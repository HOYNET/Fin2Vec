import torch
from torch.utils.data import DataLoader
from fin2vec import Fin2VecDataset
import numpy as np

inputs = ["Open", "Close", "High", "Low"]
dataset = Fin2VecDataset(
    "data/NASDAQ_FC_STK_IEM_IFO.csv",
    "data/data_2.csv",
    ["Open", "Close", "High", "Low"],
    300,
    ("2020-03-01", "2022-03-01", "%Y-%m-%d"),
    True,
)

print(dataset.__len__())

traindataLoader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True,
    pin_memory=True,
    num_workers=5
)

for i, batch in enumerate(traindataLoader):
    src, index, end = batch["src"], batch["index"], batch["end"]
    print(1)
