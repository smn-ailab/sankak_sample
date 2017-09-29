# coding:utf-8
import requests
import numpy as np
import pandas as pd
from io import StringIO
from os import path

# 調査会社から購入した女性と男性とわかっているユーザーID
female_ids = np.loadtxt(path.join(path.dirname(__file__),
                               '../client_data', 'gender_female_hash.csv'), dtype=np.uint64)
male_ids = np.loadtxt(path.join(path.dirname(__file__),
                               '../client_data', 'gender_male_hash.csv'), dtype=np.uint64)


url = "http://localhost:8080"
s = requests.session()

# 女性とわかっているユーザーの内2,000人を正例として送信。
# 男性とわかっているユーザーの内2,000人を負例として送信。
params = {"positive": ",".join(map(str, female_ids[0:2000].tolist())),
          "negative": ",".join(map(str, male_ids[0:2000].tolist())), }
r = s.post(url, data=params)

# serverに正例/負例をPOSTして、各ユーザーの予測確率(TSV)を受け取る
result = pd.read_csv(StringIO(
    r.text), sep='\t').set_index('smn_uid')

# True Positive(女性の中で女性と予測できたユーザー数 = 正例確率 >= 0.5)
TP = len(result[(result.index.isin(female_ids)) & (result.predicted >= 0.5)])

# True Negative(男性の中で男性と予測できたユーザー数 = 正例確率 < 0.5)
TN = len(result[(result.index.isin(male_ids)) & (result.predicted < 0.5)])

# レスポンスがあったユーザーの中で、正解がわかっているユーザー数
n_predicted = len(result[(result.index.isin(female_ids))
                         | result.index.isin(male_ids)])

print("TP", TP)
print("TN", TN)
print("n_predicted", n_predicted)
print("accuracy", float(TP + TN) / n_predicted)
