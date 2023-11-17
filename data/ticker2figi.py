import requests
import pandas as pd
import json
import time

ticker = pd.read_csv("./data/NASDAQ_FC_STK_IEM_IFO.csv", encoding="CP949")
ticker["FIGI"] = ""

url = "https://api.openfigi.com/v3/mapping"
data = [{"idType": "ID_ISIN", "idValue": "", "exchCode": "US"}]

for i, isin in enumerate(ticker["isin_cd"]):
    data[0]["idValue"] = isin
    response = requests.post(url, json=data).content.decode("utf-8")
    while (
        response == "Too many requests, please try again later."
        or response
        == "upstream connect error or disconnect/reset before headers. reset reason: connection termination"
    ):
        time.sleep(30)
        response = requests.post(url, json=data).content.decode("utf-8")
    response = json.loads(response)[0]
    if "data" in response:
        figi = response["data"][0]["figi"]
        ticker["FIGI"][i] = figi
    if i % 10 == 0:
        print(figi, " ", i)
        ticker.to_csv("./data/figi.csv")
