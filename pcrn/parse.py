import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class PCRNDataset(Dataset):
    def __init__(
        self,
        codeFilePath,
        priceFilePath,
        inputFeatures: list,
        outputFeatures: list,
        cp949=True,
        term: int = None,
    ):
        self.rawCode: pd.DataFrame = (
            pd.read_csv(codeFilePath, encoding="CP949")
            if cp949
            else pd.read_csv(codeFilePath)
        )
        self.rawPrice: pd.DataFrame = pd.read_csv(priceFilePath)
        self.rawPrice["Date"] = pd.to_datetime(self.rawPrice["Date"], format="%Y-%m-%d")
        self.rawPrice.sort_values(by=["Date"], inplace=True)
        self.rawPrice["tck_iem_cd"] = self.rawPrice["tck_iem_cd"].str.strip()

        self.term = term
        self.stkCnt = self.rawPrice.value_counts(["tck_iem_cd"])
        self.stkCnt = (
            self.stkCnt[self.stkCnt >= 2 * self.term].to_frame().reset_index(drop=False)
        )
        self.stkCnt["tck_iem_cd"] = self.stkCnt["tck_iem_cd"].str.strip()
        self.stkCnt["cum_cnt"] = (self.stkCnt["count"] - self.term).cumsum()

        self.rawPrice = self.rawPrice[
            self.rawPrice["tck_iem_cd"].isin(self.stkCnt["tck_iem_cd"])
        ]
        self.length = self.stkCnt.tail(1).iloc[0]["cum_cnt"]

        self.inputs = inputFeatures
        self.outputs = outputFeatures

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        stk = self.stkCnt[self.stkCnt["cum_cnt"] >= index + 1].iloc[0]
        tck, index = stk["tck_iem_cd"], stk["count"] - self.term - (
            stk["cum_cnt"] - index
        )
        raw = self.rawPrice[self.rawPrice["tck_iem_cd"] == tck].iloc[
            index : index + self.term
        ]

        data = raw[self.inputs]
        label = raw[self.outputs]
        data = data.to_numpy().transpose((1, 0))
        label = label.to_numpy().transpose((1, 0))

        return {"data": data, "label": label}
