import torch
from transformers import BertForSequenceClassification, BertTokenizer


def modelscope_predict(input):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    semantic_cls = pipeline(Tasks.text_classification, 'iic/nlp_structbert_sentiment-classification_chinese-large')
    pre_list = ['启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音',
                '我很喜欢',
                '吴总好帅',
                '海尔欠我的工钱到底什么时候才给我',
                ]
    pre_list_2 = ['启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音'] * 100
    cls = semantic_cls(input=pre_list_2)

    print(cls)
def bert_predict():
    text = '启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音'
    model = BertForSequenceClassification.from_pretrained("results/check-point-4785")
    tokenizer = BertTokenizer.from_pretrained("model/bert-base-chinese")  # 加载中文 BERT 的 tokenizer
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # 定义 ID 到标签的映射（与训练时的 label2id 对应）
    id2label = {0: "正向", 1: "中性", 2: "负向"}
    # 禁用梯度计算，加快推理速度
    with torch.no_grad():
        outputs = model(**inputs)  # 模型前向推理

    # 获取模型输出的 logits（未归一化的预测分数）
    logits = outputs.logits

    # 取 logits 最大值对应的索引作为预测类别 ID
    predicted_class = logits.argmax().item()

    # 返回原始文本和预测的情感类别标签
    return {"text": text, "sentiment": id2label[predicted_class]}

if __name__ == '__main__':
    print(bert_predict())



