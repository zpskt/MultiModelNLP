# 前言
使用bert-base-chinese模型进行情感分析，模型训练成本低，不需要大量的资源。
目前的方式是调用modelscope的模型，模型的准确率可能会受到环境、数据等因素的影响。

为了模型落地，使用接口的形式提供服务，方便各个平台统一调用。
优点：
1. 统一接口，方便平台调用
2. 模型训练成本低，不需要大量的资源
缺点：
1. 模型是针对特定的场景，不能通用
2. 模型的准确率可能会受到环境、数据等因素的影响

### 准备
1. 使用conda方式
python = 3.12.11
```shell
conda create -n sentiment python=3.12.11
conda activate sentiment
pip install -r requirements.txt
```
2. 原生python环境
python = 3.12.11
```shell
pip install -r requirements.txt
```

### 使用
运行api服务
```shell
cd app
uvicorn sentiment_analysis_api:app --reload
```
api调用 建议批次100个执行
详情使用参照 app/test_sentiment_analysis_api.py



# 常用命令
## 备份环境
```shell
pip install pip-chill
pip-chill > requirements.txt
```
## 查看是否支持cuda
```shell
python -c "import torch; print(torch.mps.is_available())"
```
## 下载模型
```shell
cd model/bert-base-chinese
wget https://hf-mirror.com/google-bert/bert-base-chinese/resolve/main/pytorch_model.bin
```