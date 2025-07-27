# 前言
NLP多模型使用。
目前提供三种NLP场景服务：
1、中文情感二分类：提供Api接口化服务，输入文本，输出情感二分类结果。
2、中文文本多分类：提供Api接口化服务，根据训练数据，对输入文本进行多分类
3、中文搭建知识库LLM模型：提供Api接口化服务。

## 整体环境安装
本项目统一使用Anaconda管理环境
### 安装并配置镜像源
请确保你已经安装了**conda**
```shell
# 清华大学镜像站
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

# 中科大镜像站
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/

# 北京外国语大学镜像站
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/free/

```
下载项目
```shell
git clone https://github.com/zpskt/MultiModelNLP.git
cd MultiModelNLP
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

## 中文情感二分类
使用[魔塔社区](https://www.modelscope.cn/models/iic/nlp_structbert_nli_chinese-large)模型，使用modelscope提供的模型进行推理
### 环境
创建环境
```shell
conda create -n sentiment --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.12.11
```
安装依赖
```shell
conda activate sentiment
pip install -r src/sentiment/requirements.txt
#pip install -r src/sentiment/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```
激活环境
```shell
conda activate sentiment
```
### 使用
运行Api服务
```shell
cd src/sentiment/
uvicorn api:app --reload
```
api调用示例
```shell
python src/sentiment/test_sentiment_analysis_api.py
```

## 中文文本多分类
### 环境
创建环境
```shell
conda create -n sentiment --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.12.11
```
安装依赖
```shell
conda activate sentiment
pip install -r src/bert/requirements.txt
#pip install -r src/bert/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```
激活环境
```shell
conda activate sentiment
```
预训练模型通过huggingface下载
准备训练数据 :将训练数据放置于data/train.csv 

配置huggingface镜像
```shell
export HF_ENDPOINT=https://hf-mirror.com
```

开始训练
```shell
python train.py 
```
如果下载失败，那么就手动下载模型
```shell
wget -P model/bert-base-chinese https://hf-mirror.com/google-bert/bert-base-chinese/resolve/main/pytorch_model.bin
```
查看训练结果 
训练结果放置于results文件夹下 
模型检查点目录结构

| 文件名              | 说明                                                         |
|---------------------|--------------------------------------------------------------|
| [config.json](file://D:\zpskt\sentiment\model\bert-base-chinese\config.json)       | 模型配置文件，保存模型的超参数和架构配置信息   |              |
| `model.safetensors` | 模型权重文件，使用 safetensors 格式存储模型参数              |
| `optimizer.pt`      | 优化器状态文件，保存优化器的参数和状态，用于恢复训练         |
| `rng_state.pth`     | 随机数生成器状态文件，确保训练过程的可重现性                 |
| `scheduler.pt`      | 学习率调度器状态文件，保存学习率调整策略的状态               |
| `trainer_state.json`| 训练器状态文件，记录训练过程中的各种状态信息                 |
| `training_args.bin` | 训练参数文件，保存训练时使用的命令行参数配置                 |

该目录保存了训练过程中的模型检查点，包含模型权重、配置和训练状态等文件
用于模型的恢复训练或推理部署
当使用时，加载模型选择某个文件夹模型即可，要保证结构与我的一致
### 使用
运行Api服务
```shell
cd src/bert
uvicorn api:app --reload
```

## 中文搭建知识库LLM模型
faiss+llm的形式构建。
doc_file:知识文档存放路径
faiss_index：索引文件存放路径
## 环境
创建环境
```shell
conda create -n llm-faiss --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.9
```
安装依赖
```shell
conda activate llm-faiss
conda install langchain -c conda-forge
pip install -U langchain-community langgraph langchain-anthropic tavily-python langgraph-checkpoint-sqlite
pip install -U sentence-transformers
conda install  docx2txt
pip install faiss-cpu
#pip install -r src/llm/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
``` 
激活环境
```shell
conda activate llm-faiss
```
## 使用
运行Api服务
```shell
cd src/llm
uvicorn api:app --reload
```
程序使用方式
```shell
python src/llm/main.py
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
