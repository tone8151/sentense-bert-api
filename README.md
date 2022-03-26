# sentense-bert  API

予め用意しておいた想定問答集にマッチする質問を投げればその答えが返ってくるAPIです

# Requirement

* Docker
* docker-compose

※ それぞれのインストール方法は割愛

# Installation
## このリポジトリをcloneする
```bash
git clone git@github.com:tone8151/sentense-bert-api.git
```
## 学習済みモデルをプロジェクト直下に配置
modelディレクトリをそのまま直下に配置する

## イメージ作成、コンテナ起動
プロジェクト直下に移動し、
```bash
docker-compose up -d
```
## コンテナの確認
```bash
docker ps -a
```
docker_sentense_bert_appコンテナのstatusがUpになっていればOK

## APIサーバーが起動しているか確認
"http://localhost:8000"でHello, worldが表示されればOK

## コンテナの停止
```bash
docker-compose down
```
