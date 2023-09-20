import pandas as pd
from torch.utils.data import Dataset


class stockDataset(Dataset):
    def __init__(
        self,
        codeFilePath,
        priceFilePath,
        cp949=True,
        term: (int, int) = None,
    ):
        self.rawCode: pd.DataFrame = (
            pd.read_csv(codeFilePath, encoding="CP949")
            if cp949
            else pd.read_csv(codeFilePath)
        )
        self.rawPrice: pd.DataFrame = pd.read_csv(priceFilePath)
        self.stockCode: pd.Series = self.rawCode["tck_iem_cd"]
        self.columns = self.rawPrice.columns.drop(labels=["trd_dt", "tck_iem_cd"])
        self.term = term

    def __len__(self):
        return len(self.stockCode)

    def __getitem__(self, index):
        code: str = self.stockCode.iloc[index].strip()
        data: pd.DataFrame = self.rawPrice[self.rawPrice["tck_iem_cd"] == code].copy()
        data["trd_dt"] = pd.to_datetime(data["trd_dt"], format="%Y%m%d")
        data.set_index("trd_dt", inplace=True)
        data.drop(
            columns=[
                "tck_iem_cd",
                "gts_sll_cns_sum_qty",
                "gts_byn_cns_sum_qty",
                "gts_acl_trd_qty",
                "gts_iem_end_pr",
            ],
            inplace=True,
        )
        if self.term:
            data = data.iloc[self.term[0] : self.term[1]]
        data = data.to_numpy().transpose((1, 0))
        return {"data": data, "label": data}
