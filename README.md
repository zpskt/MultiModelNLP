#开始

## 创建环境
conda env create -f environment.yml
## 将conda环境导出
```shell
conda env export > environment.yml
```
## 查看是否支持cuda
```shell
python -c "import torch; print(torch.mps.is_available())"
```
## 下载模型
```shell
cd model/bert-base-chinese
wget https://hf-mirror.com/google-bert/bert-base-chinese/resolve/main/pytorch_model.bin?download=true
```