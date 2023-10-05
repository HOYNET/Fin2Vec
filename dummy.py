import torch
from torch.utils.data import DataLoader
from fin2vec import Fin2VecDataset

dataset = Fin2VecDataset(
    "data/NASDAQ_FC_STK_IEM_IFO.csv",
    "data/data_2.csv",
    ["Open", "Close", "High", "Low"],
    300,
    ("2020-03-01", "2022-03-01", "%Y-%m-%d"),
    True,
)

traindataLoader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True,
)

for i, batch in enumerate(traindataLoader):
    src, index = batch["src"], batch["index"]
    print(src)
