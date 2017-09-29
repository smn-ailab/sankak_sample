# 動作方法

## ダウンロード＆解凍

サンプルコードを実行するためには、以下のファイルを解凍する必要が有ります。

```sh
$ cd server_data
$ unzip data.zip
$ cd ../client_data
$ unzip data.zip
```

## 実行方法

```sh
$ python server/server.py &
$ python client/client.py
```

## 注意点

* サーバーは8080ポートで起動します。チーム内でバッティングしないようにポートを変更してください。
