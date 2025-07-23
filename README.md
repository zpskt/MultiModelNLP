# 前言
提供两种方式提供情感分析服务：
1. 通用化服务：通过api方式提供服务，情感分类仅支持 正向 负向 双分类，且由于训练数据集的领域化，可能针对特殊场景，效果不理想。
2. 定制化服务：使用谷歌提出的bert-base-chinese作为预训练模型，针对自己的特殊场景，对模型定制化训练，可以实现情感多分类，且效果良好。

## 环境安装
### 安装
请确保你已经安装了**conda**
1. 下载项目
```shell
git clone https://github.com/zpskt/sentiment.git
cd sentiment
```
2. 创建环境
```shell
conda create -n sentiment --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.12.11
```
3. 安装依赖
```shell
conda activate sentiment
pip install -r requirements.txt
#pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```
4. 使用cuda
查看cuda是否支持
```shell
python -c "import torch; print(torch.mps.is_available())"
```
去[官网](https://pytorch.org/get-started/locally/)获取cuda版本安装 （注意需要看你的cuda版本，原则向下兼容）
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 通用化服务
使用[魔塔社区](https://www.modelscope.cn/models/iic/nlp_structbert_nli_chinese-large)模型，使用modelscope提供的模型进行推理

### 使用
运行Api服务
```shell
uvicorn app/sentiment_analysis_api:app --reload
```
api调用示例
```shell
python app/test_sentiment_analysis_api.py
```
## 定制化服务
预训练模型通过huggingface下载
1. 准备训练数据
将训练数据放置于data/train.csv
2. 下载预训练模型
```shell
wget -P model/bert-base-chinese https://hf-mirror.com/google-bert/bert-base-chinese/resolve/main/pytorch_model.bin
```
或者配置huggingface镜像
```shell
export HF_ENDPOINT=https://hf-mirror.com
```

2.开始训练
```shell
python train.py 
```
3. 查看训练结果
训练结果放置于results文件夹下
### 使用
运行Api服务
```shell
```
api调用示例
```shell
```

# 常用命令
## 备份环境
```shell
pip install pip-chill
pip-chill > requirements.txt
conda list --explicit > requirements.txt
```
## 查看是否支持cuda
```shell
python -c "import torch; print(torch.cuda.is_available())" #linux
python -c "import torch; print(torch.mps.is_available())" #macos
```
## 查看系统GPU占用
```shell
  nvidia-smi
```
