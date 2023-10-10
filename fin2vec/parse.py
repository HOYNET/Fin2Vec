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
        ngroup: int = None,
    ):
        self.stock_info_dtype = np.dtype(
            [
                ("code", "U10"),  # 최대 10글자의 유니코드 문자열
                ("index", "i4"),  # 4바이트 정수
                ("minTime", "M8[ns]"),  # 나노초 정밀도의 날짜 시간
                ("maxTime", "M8[ns]"),  # 나노초 정밀도의 날짜 시간
                ("length", "i4"),  # 4바이트 정수
            ]
        )
        self.loadFromFile(codePath, pricePath, cp949)
        self.periodInspect(period, term)
        self.inputs = inputs
        self.timeline: np.ndarray = np.array(
            self.rawPrice["Date"].drop_duplicates(), dtype=np.datetime64
        )
        self.timeline.sort()

        self.groups(ngroup)

    def __len__(self):
        return self.ngroup

    def __getitem__(self, index):
        msk, self.minTime, self.maxTime = (
            self.groupMsk[index],
            self.groupBegin[index],
            self.groupEnd[index],
        )  # msk should be inversed

        src = np.zeros(
            shape=(
                self.length,
                len(self.inputs),
                self.term,
            )
        )
        _srcDF = self.rawPrice[
            self.rawPrice["tck_iem_cd"].isin(self.stockCode[msk])
        ].groupby("tck_iem_cd", sort=True)
        srcDF = (
            _srcDF.apply(self.tighten)
            .reset_index(drop=True)
            .groupby("tck_iem_cd", sort=True)
        )

        _src = np.array([group[self.inputs].values for _, group in srcDF])

        src[msk] = src[msk] + _src.transpose((0, 2, 1))
        indices = np.random.permutation(np.arange(self.length, dtype=np.int32))
        return {"src": src[indices], "index": indices, "mask": msk}

    def loadFromFile(self, codePath: str, pricePath: str, cp949: bool = True):
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

        groups = self.rawPrice.groupby("tck_iem_cd")
        lengths = groups.size()
        adjusted_lengths = lengths.reindex(self.stockCode).fillna(0).astype(int).values
        valid_codes = adjusted_lengths >= self.term
        self.stockCode = self.stockCode[valid_codes].sort_values()
        self.rawPrice = (
            self.rawPrice[self.rawPrice["tck_iem_cd"].isin(self.stockCode)]
            .sort_values(by="tck_iem_cd")
            .sort_values(by="Date")
        )

        groups = self.rawPrice.groupby("tck_iem_cd")

        self.infos = np.array(
            [
                (
                    x[0],
                    i,
                    x[1]["Date"].min(),
                    x[1]["Date"].max(),
                    len(x[1]),
                )
                for i, x in enumerate(groups)
            ],
            dtype=self.stock_info_dtype,
        )
        self.length = len(self.stockCode)

    def groups(self, ngroup: int):
        self.ngroup = ngroup

        self.groupMsk = np.random.random_integers(
            size=(self.ngroup, self.length), low=0, high=1
        ).astype(np.bool_)

        startPos = np.random.random_integers(
            low=0, high=len(self.timeline) - self.term - 1, size=(self.ngroup)
        )
        self.groupBegin = self.timeline[startPos]
        self.groupEnd = self.timeline[startPos + self.term - 1]
        self.groupMsk = (
            (
                np.tile(self.groupBegin, (len(self.infos["minTime"]), 1)).transpose(
                    -1, -2
                )
                >= np.tile(self.infos["minTime"], (self.ngroup, 1))
            )
            & (
                np.tile(self.groupEnd, (len(self.infos["maxTime"]), 1)).transpose(
                    -1, -2
                )
                <= np.tile(self.infos["maxTime"], (self.ngroup, 1))
            )
            & self.groupMsk
        )

    def tighten(self, prices: pd.DataFrame) -> pd.DataFrame:
        result = prices[
            (prices["Date"] >= self.minTime) & (prices["Date"] <= self.maxTime)
        ].reset_index(drop=True)

        if len(result) != self.term:
            return pd.DataFrame(np.zeros(result.shape), columns=result.columns.copy())
        else:
            return result
