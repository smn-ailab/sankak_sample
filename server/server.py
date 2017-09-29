# coding:utf-8
from bottle import route, run, template, request, post
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
from os import path

@post('/')
def fit_predict():
    positive_ids = list(map(int, request.forms.get('positive').split(','))) #正例IDs
    negative_ids = list(map(int, request.forms.get('negative').split(','))) #負例IDs

    # 特徴量行列ファイル(CSR形式)の読み出し
    csr_file = path.join(path.dirname(__file__), "../server_data", "extracted_features")
    smnuid = np.loadtxt(csr_file + ".smnuid_hash", dtype=np.uint64)
    data = csr_matrix((
        np.loadtxt(csr_file + ".data", dtype=np.float),
        np.loadtxt(csr_file + ".indices", dtype=np.uint64),
        np.loadtxt(csr_file + ".indptr", dtype=np.uint64)),
        shape=(len(smnuid),  57980))

    # clientから指定された正負例ユーザーの中で、特徴量行列ファイルに含まれるユーザーだけ取り出す。
    train_X = vstack([data[np.isin(smnuid, positive_ids)],
                data[np.isin(smnuid, negative_ids)]])

    # 少なすぎると精度が悪くなるので、一応確認
    # POSTされた正例/負例ユーザーがサーバーサイドで保持している特徴量DBに存在しない可能性有り
    n_positive = np.isin(smnuid, positive_ids).sum()
    n_negative = np.isin(smnuid, negative_ids).sum()
    print("n_positive", n_positive)
    print("n_negative", n_negative)

    # 正例:1, 負例:0として ベクトルを用意
    train_Y = np.concatenate([np.ones(n_positive), np.zeros(n_negative)], axis=0)

    # ロジスティック回帰の学習。
    lr = LogisticRegression()
    lr.fit(train_X, train_Y)

    # ユーザーIDと予測結果をTSV形式でレスポンス
    result = pd.DataFrame(smnuid, columns=['smn_uid']).set_index('smn_uid')
    result = result.assign(predicted=lr.predict_proba(data)[:,1])
    return result.to_csv(sep='\t')

run(host='localhost', port=8080)
