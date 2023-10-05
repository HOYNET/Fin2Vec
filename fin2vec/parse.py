import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime


class Fin2VecDataset(Dataset):
    def __init__(
        self,
        codePath: str,
        pricePath: str,
        inputs: list,
        term: int,
        period: (str, str, str),
        cp949=True,
        range: (int, int) = None,
    ):
        self.loadFromFile(codePath, pricePath, cp949)
        self.periodInspect(period, term)
        self.inputs = inputs
        self.inputs.append("tck_iem_cd")
        self.inputs.append("Date")

        if range:
            self.range: (int, int) = range
        else:
            self.range: (int, int) = (1, self.length)
        assert self.range[0] < self.range[1] and self.range[1] <= self.length
        self.timeline = self.rawPrice["Date"].drop_duplicates()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        tgtCode = self.stockCode.iloc[index]
        tgt = self.rawPrice[self.rawPrice["tck_iem_cd"] == tgtCode].copy()

        minTime, maxTime = tgt["Date"].min(), tgt["Date"].max()
        tgtTerm = (maxTime - minTime).days
        timeline = self.timeline[self.timeline >= minTime]
        timeline = timeline[timeline < maxTime]
        tgtTerm = len(timeline)
        startIdx = np.random.randint(low=0, high=tgtTerm - self.term)
        endIdx = startIdx + self.term
        startTime, endTime = timeline.iloc[startIdx], timeline.iloc[endIdx]
        rawData: pd.DataFrame = self.rawPrice.loc[
            self.rawPrice["Date"] >= startTime, self.inputs
        ]
        rawData = rawData[rawData["Date"] < endTime]
        codes, lengths = (
            rawData["tck_iem_cd"].drop_duplicates(),
            rawData.groupby("tck_iem_cd").size(),
        )
        lengths = lengths.reindex(codes).fillna(0).astype(int)
        codes = lengths[lengths == self.term].index
        src = [
            pd.DataFrame(x[1])
            .sort_values(by=["Date"], ascending=True)
            .drop(columns=["Date", "tck_iem_cd"])
            .to_numpy()
            .transpose(-1, -2)
            for x in rawData[rawData["tck_iem_cd"].isin(codes)].groupby("tck_iem_cd")
        ]
        src = np.array(src)
        index = np.array(self.stockCode[self.stockCode.isin(codes)].index)
        if len(src) != self.length:
            gap, shape = self.length - len(src), src.shape
            padsrc, padindex = (
                np.zeros((gap, shape[1], shape[2])),
                np.zeros((gap), dtype=np.int64) - 1,
            )
            src, index = np.concatenate([src, padsrc], axis=0), np.concatenate(
                [index, padindex], axis=0
            )

        return {"src": src, "index": index}

    def loadFromFile(self, codePath, pricePath, cp949):
        self.rawCode: pd.DataFrame = (
            pd.read_csv(codePath, encoding="CP949") if cp949 else pd.read_csv(codePath)
        )
        self.rawPrice: pd.DataFrame = pd.read_csv(pricePath)
        self.rawPrice = self.rawPrice.drop(columns=["Unnamed: 0"])
        self.stockCode: pd.Series = self.rawCode["tck_iem_cd"].str.strip()
        self.length = len(self.stockCode)

    def periodInspect(self, period, term):
        self.dateFormat = period[2]
        self.starttime = datetime.strptime(period[0], self.dateFormat)
        self.endtime = datetime.strptime(period[1], self.dateFormat)
        self.termInspect(term)
        self.rawPrice["Date"] = pd.to_datetime(
            self.rawPrice["Date"], format=self.dateFormat
        )

    def termInspect(self, term):
        self.term, self.period = term, (self.endtime - self.starttime).days
        assert self.term < self.period
        lengths = self.rawPrice.groupby("tck_iem_cd").size()
        adjusted_lengths = lengths.reindex(self.stockCode).fillna(0).astype(int).values
        valid_codes = adjusted_lengths >= self.term
        self.stockCode = self.stockCode[valid_codes]
        self.rawPrice = self.rawPrice[self.rawPrice["tck_iem_cd"].isin(self.stockCode)]
        self.length = len(self.stockCode)
