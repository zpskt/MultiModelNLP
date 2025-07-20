# 导入必要的库
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import re

# 加载 JSON 格式的评论数据，假设文件名为 comments.json
# 1️⃣ 加载 CSV 数据，并只保留 content 和 sentiment 两列
data = pd.read_csv("comments.csv")[["content", "sentiment"]]
# 2️⃣ 重命名列名为 text 和 label（适配后续流程）
data.columns = ["text", "label"]
# 将 pandas DataFrame 转换为 HuggingFace Dataset 格式
dataset = Dataset.from_pandas(data)

# 定义标签映射：将文本标签映射为模型训练所需的数字 ID
label2id = {"正向": 0, "中性": 1, "负向": 2}
# 使用 map 方法将每个样本的 label 字段转换为对应的数字 ID
dataset = dataset.map(lambda x: {"labels": label2id[x["label"]]})

#  3️⃣ 添加清洗函数
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # 去除 URL
    text = re.sub(r'[@#]\S+', '', text)                # 去除 @ 和 #
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text) # 去除表情符号
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)  # 保留中英文和数字
    text = re.sub(r'\s+', ' ', text).strip()           # 去除多余空格
    return text

# 对文本进行清洗
dataset = dataset.map(lambda x: {"text": clean_text(x["text"])})

# 过滤掉空文本
dataset = dataset.filter(lambda x: x["text"].strip() != "")

# 加载中文 BERT 的 tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 定义分词函数，对文本进行 tokenization、padding 和 truncation
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 对整个数据集进行分词处理，batched=True 表示批量处理
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# 拆分数据集为训练集和验证集（例如 90% 训练，10% 验证）
split_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]



# 加载预训练的 BERT 模型，并设置 num_labels=3 表示这是一个三分类任务
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=3)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",               # 模型输出保存路径
    evaluation_strategy="epoch",           # 每个 epoch 结束后进行评估
    learning_rate=2e-5,                    # 学习率
    per_device_train_batch_size=16,        # 每个设备的训练 batch size
    num_train_epochs=3,                    # 总共训练的 epoch 数
    weight_decay=0.01,                     # 权重衰减（L2 正则化）
)

# 创建 Trainer，用于管理训练过程
trainer = Trainer(
    model=model,                           # 要训练的模型
    args=training_args,                    # 训练参数
    train_dataset=train_dataset,      # 训练数据集
    eval_dataset=eval_dataset,       # 评估数据集（这里使用训练集本身作为验证集）
)

# 开始训练模型
trainer.train()
