# AWS デプロイガイド (Deployment Guide)

このガイドでは、ナンクロSolverをAWS環境（Lambda, API Gateway）にデプロイする手順を説明します。

## 前提条件 (Prerequisites)

以下のツールがインストールされ、パスが通っていることを確認してください。

1.  **Node.js & npm**: `node --version`, `npm --version`
2.  **Python 3.10+**: `python --version`
3.  **Docker**: `docker --version` (Docker Desktop等が起動していること)
4.  **AWS CLI**: `aws --version` (認証情報 `aws configure` が設定済みであること)

## デプロイ手順

### 1. AWS CDK のインストール
まだインストールしていない場合は、AWS CDKをグローバルインストールします。

```bash
npm install -g aws-cdk
```

### 2. 依存パッケージのインストール
インフラ構築用のPythonライブラリをインストールします。

```bash
# プロジェクトルートで実行
pip install -r requirements_infra.txt
```

### 3. CDKの初期設定 (Bootstrapping)
初めてCDKを使用するAWSアカウント・リージョンの場合、Bootstrapが必要です。
（すでにS3バケット等が作成されている場合はスキップ可能です）

```bash
cd infra
cdk bootstrap
```

### 4. デプロイの実行
`infra` ディレクトリ内でデプロイコマンドを実行します。
これにより、ECRリポジトリの作成、Dockerイメージのビルド＆プッシュ、Lambda関数とAPI Gatewayの作成が自動的に行われます。

```bash
# infraディレクトリにいることを確認
cdk deploy
```

途中で「Do you wish to deploy these changes (y/n)?」と聞かれるので、`y` を入力してEnterを押してください。

### 5. デプロイ完了と確認
デプロイが成功すると、ターミナルに以下のような出力が表示されます。

```text
Outputs:
NankuroSolverStack.ApiUrl = https://xxxxxxxxxx.execute-api.ap-northeast-1.amazonaws.com/
NankuroSolverStack.LambdaArn = arn:aws:lambda:ap-northeast-1:123456789012:function:NankuroSolverStack...
```

**`ApiUrl`** が、作成されたAPIのエンドポイントです。このURLをコピーしてください。

## フロントエンドの設定

デプロイされたAPIを利用するようにフロントエンドの設定を変更します。

1.  `frontend/result_solve.html` を開きます。
2.  Javascript内の定数 `API_URL` を、上記で取得した `ApiUrl` + `/api/solve` に書き換えます。

```javascript
// 例
const API_URL = "https://xxxxxxxxxx.execute-api.ap-northeast-1.amazonaws.com/api/solve";
```

これで、フロントエンドからAWS上のSolver（課金ロジック付き）を利用できるようになります。

## クリーンアップ (削除)

リソースを削除（破棄）したい場合は、以下のコマンドを実行します。

```bash
cd infra
cdk destroy
```
