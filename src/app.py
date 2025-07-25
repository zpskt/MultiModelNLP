from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException  # 引入 HTTPException 用于错误处理
from flask import Flask, jsonify
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pydantic import BaseModel

from llm.document_loader import DocumentProcessor
from llm.qa_system import QASystem

app = Flask(__name__)
# 初始化情感分类模型
semantic_cls = pipeline(Tasks.text_classification, 'iic/nlp_structbert_sentiment-classification_chinese-large')

# 初始化问答系统
processor = DocumentProcessor()
vector_store = processor.load_vector_store("faiss_index")

# 定义情感分析请求体结构
class PredictRequest(BaseModel):
    text: List[str]

# 定义问答请求体结构
class QARequest(BaseModel):
    question: str

# 定义添加文档请求体结构
class AddDocumentsRequest(BaseModel):
    file_paths: List[str]

@app.route('/llm/ask/', methods=['POST'])
def ask_question(request: QARequest):
    """
    处理问答请求
    """
    question = request.question

    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    # 初始化问答系统实例
    qa_system = QASystem(vector_store)
    
    # 获取答案
    result = qa_system.ask(question)
    
    return jsonify(result)

# 添加新文档到向量存储的接口
@app.route('/llm/add_documents/', methods=['POST'])
def add_documents(request: AddDocumentsRequest):
    """
    向现有向量存储中添加新文档
    """
    global vector_store
    
    try:
        # 添加新文档到现有向量存储
        vector_store = processor.add_documents_to_store(vector_store, request.file_paths)
        
        # 保存更新后的向量存储
        processor.save_vector_store(vector_store, "faiss_index")
        
        return jsonify({
            "status": "success",
            "message": f"Successfully added documents to vector store"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    try:
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
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)