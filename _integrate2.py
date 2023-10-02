import pandas as pd

data1: pd.DataFrame = pd.read_csv(
    "/home/lhstar/SKKU/NHContest/crawl_5/combined_final_stock_data.csv"
)
data2: pd.DataFrame = pd.read_csv(
    "/home/lhstar/SKKU/NHContest/crawl_5/combined_final_stock_data2.csv"
)
data3: pd.DataFrame = pd.read_csv(
    "/home/lhstar/SKKU/NHContest/crawl_5/combined_final_stock_data3.csv"
)
data4: pd.DataFrame = pd.read_csv(
    "/home/lhstar/SKKU/NHContest/crawl_5/combined_final_stock_data4.csv"
)
data5: pd.DataFrame = pd.read_csv(
    "/home/lhstar/SKKU/NHContest/crawl_5/combined_final_stock_data5.csv"
)

data: pd.DataFrame = pd.concat([data1, data2, data3, data4, data5], axis=0)

data["tck_iem_cd"] = data["tck_iem_cd"].str.strip()
data["Date"] = pd.to_datetime(data["Date"])
print(data[data["tck_iem_cd"] == "AACG"])

print(data.isna().any())
data = data.drop_duplicates()
data.to_csv("./data/data_2.csv")
