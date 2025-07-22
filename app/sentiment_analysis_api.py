# 导入必要的库
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException  # 引入 HTTPException 用于错误处理
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pydantic import BaseModel

app = FastAPI()

# 定义请求体结构
class PredictRequest(BaseModel):
    text: List[str]

@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    try:
        # 初始化模型
        semantic_cls = pipeline(Tasks.text_classification, 'iic/nlp_structbert_sentiment-classification_chinese-large')

        # 调用模型预测
        cls = semantic_cls(input=request.text)

        # 将输入文本与预测结果合并
        results = [
            {
                "text": text,
                "scores": item["scores"],
                "labels": item["labels"]
            }
            for text, item in zip(request.text, cls)
        ]

        return {
            "code": 200,
            "status": "success",
            "sentiment": results
        }

    except Exception as e:
        # 捕获异常并返回错误状态
        return {
            "code": 500,
            "status": "error",
            "message": str(e),
            "sentiment": []
        }
