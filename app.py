# 导入必要的库
from fastapi import FastAPI  # FastAPI 是一个现代、快速的 Web 框架，用于构建 API
from transformers import BertTokenizer, BertForSequenceClassification  # 用于加载 BERT 分词器和分类模型
import torch  # PyTorch，用于深度学习推理

# 创建 FastAPI 实例
app = FastAPI()

# 从指定路径加载训练好的模型和对应的 tokenizer
# "./results" 是训练脚本中 Trainer 保存模型的目录
model = BertForSequenceClassification.from_pretrained("results")
tokenizer = BertTokenizer.from_pretrained("model/bert-base-chinese")  # 加载中文 BERT 的 tokenizer

# 定义 ID 到标签的映射（与训练时的 label2id 对应）
id2label = {0: "正向", 1: "中性", 2: "负向"}

# 定义一个 POST 接口，路径为 /predict，用于接收文本并返回情感预测
@app.post("/predict")
def predict(text: str):
    # 使用 tokenizer 对输入文本进行编码，转换为模型可接受的输入格式（input_ids, attention_mask 等）
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # 禁用梯度计算，加快推理速度
    with torch.no_grad():
        outputs = model(**inputs)  # 模型前向推理

    # 获取模型输出的 logits（未归一化的预测分数）
    logits = outputs.logits

    # 取 logits 最大值对应的索引作为预测类别 ID
    predicted_class = logits.argmax().item()

    # 返回原始文本和预测的情感类别标签
    return {"text": text, "sentiment": id2label[predicted_class]}
