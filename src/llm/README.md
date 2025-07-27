# 前言
langchain+faiss+llm
## 项目结构
```shell

│   ├── llm
│   │   ├── api.py
│   │   ├── document_loader.py
│   │   ├── main.py
│   │   ├── qa_system.py
│   │   └── processed_files.json
```
README.md - 项目说明文档
document_loader.py - 文档加载和处理模块
main.py - 主程序入口
qa_system.py - 问答系统实现模块
api.py - api服务模块
processed_files.json - 已处理的文件列表

## 环境安装
### 安装
请确保你已经安装了**conda**
1. 下载项目
```shell
git clone https://github.com/zpskt/sentiment.git
cd sentiment
```
去申请api
https://dashscope.console.aliyun.com/overview 获取apikey
2. 创建环境
```shell
conda create -n llm-faiss --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.9
```
3. 安装依赖
```shell
conda activate llm-faiss
conda install langchain -c conda-forge
pip install -U langchain-community langgraph langchain-anthropic tavily-python langgraph-checkpoint-sqlite
pip install -U sentence-transformers
conda install  docx2txt
pip install faiss-cpu
#pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```
