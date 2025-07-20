# 导入必要的库
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel
from datasets import Dataset
import pandas as pd
import re
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 设置代理 官方地址无法访问
BERT_PATH = "/Users/zhangpeng/Desktop/zpskt/sentiment/model/bert-base-chinese"
# 加载 JSON 格式的评论数据，假设文件名为 comments.json
# 1️⃣ 加载 CSV 数据，并只保留 content 和 sentiment 两列
data = pd.read_csv(
    "data/train.tsv",
    sep='\t',
    usecols=[1, 2],
    names=['label', 'text'],
    header=None,
    skiprows=1  # 跳过第一行（原表头）
)# 查看前几行确认数据结构
print(data.head())
unique_labels = data['label'].unique()

# 查看 label 列的值及其出现次数
label_counts = data['label'].value_counts()

print("Unique labels:", unique_labels)
print("\nLabel counts:\n", label_counts)
dataset = Dataset.from_pandas(data)

# 如果 label 是字符串，先转为整数
# dataset = dataset.map(lambda x: {"label": int(x["label"])})
print(next(iter(dataset)))# 查看 label 列的所有唯一值
# 直接使用原始 label（0 和 1）
dataset = dataset.map(lambda x: {"labels": int(x["label"])})

# 在 dataset.map 之后转换为 pandas 查看 label 分布
df = dataset.to_pandas()
print(df['labels'].value_counts())


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
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
# 定义分词函数，对文本进行 tokenization、padding 和 truncation
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 对整个数据集进行分词处理，batched=True 表示批量处理
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# 拆分数据集为训练集和验证集（例如 90% 训练，10% 验证）
split_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 加载预训练的 BERT 模型，并设置 num_labels=2 表示这是一个二分类任务
model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=2)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="results",               # 模型输出保存路径
    # 每个 epoch 结束后进行评估
    learning_rate=2e-5,                    # 学习率
    per_device_train_batch_size=16,        # 每个设备的训练 batch size
    num_train_epochs=3,                    # 总共训练的 epoch 数
    weight_decay=0.01,                     # 权重衰减（L2 正则化）
    use_cpu=False
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
