import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class stockDataset(Dataset):
    def __init__(
        self,
        codeFilePath,
        priceFilePath,
        cp949=True,
        term: int = None,
    ):
        self.rawCode: pd.DataFrame = (
            pd.read_csv(codeFilePath, encoding="CP949")
            if cp949
            else pd.read_csv(codeFilePath)
        )
        self.rawPrice: pd.DataFrame = pd.read_csv(priceFilePath)
        self.rawPrice = self.rawPrice.drop(columns=["Unnamed: 0"])
        self.stockCode: pd.Series = self.rawCode["tck_iem_cd"].str.strip()
        self.term = term
        self.length = len(self.stockCode)

        lengths = self.rawPrice.groupby("tck_iem_cd").size()
        adjusted_lengths = lengths.reindex(self.stockCode).fillna(0).astype(int).values
        self.cache = np.column_stack(
            (np.zeros_like(adjusted_lengths), adjusted_lengths)
        )

        valid_codes = self.cache[:, 1] >= self.term
        self.stockCode = self.stockCode[valid_codes]
        self.rawPrice = self.rawPrice[self.rawPrice["tck_iem_cd"].isin(self.stockCode)]
        self.cache = self.cache[valid_codes]
        self.length = len(self.stockCode)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        code: str = self.stockCode.iloc[index].strip()
        data: pd.DataFrame = self.rawPrice[self.rawPrice["tck_iem_cd"] == code].copy()
        data["trd_dt"] = pd.to_datetime(data["trd_dt"], format="%Y-%m-%d")
        data.set_index("trd_dt", inplace=True)
        data.drop(
            # gts_iem_ong_pr,gts_iem_hi_pr,gts_iem_low_pr,gts_iem_end_pr,gts_acl_trd_qty
            columns=[
                "gts_iem_ong_pr",
                "tck_iem_cd",
                "gts_iem_end_pr",
            ],
            inplace=True,
        )
        if self.term:
            current, maxidx = (
                np.random.randint(self.cache[index][1] - self.term + 1),
                self.cache[index][1],
            )
            new = current + self.term
            if new > maxidx:
                current, new = 0, self.term
            data = data.iloc[current:new]
        data = data.to_numpy().transpose((1, 0))
        return {"data": data, "label": data[:2]}
