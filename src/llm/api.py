import os
from typing import List

from flask import Flask, jsonify, request
from pydantic import BaseModel

from document_loader import DocumentProcessor
from qa_system import QASystem
from src.util.log import LoggerManager
log = LoggerManager().get_logger()
app = Flask(__name__)
# 初始化问答系统
processor = DocumentProcessor()
# 获取当前文件的绝对路径
# 检查向量存储是否存在，如果不存在则创建一个空的
vector_store_path = "llm/faiss_index"
if os.path.exists(vector_store_path) and os.path.exists(os.path.join(vector_store_path, "index.faiss")):
    vector_store = processor.load_vector_store(vector_store_path)
    # 初始化问答系统实例
    qa_system = QASystem(vector_store)
else:
    log.info("向量存储不存在，将在添加文档时创建新的")
    vector_store = None

# 定义问答请求体结构
class QARequest(BaseModel):
    question: str

# 定义添加文档请求体结构
class AddDocumentsRequest(BaseModel):
    file_paths: List[str]

@app.route('/llm/ask/', methods=['POST'])
def ask_question():
    """
    处理问答请求
    """
    try:
        # 从请求体中获取JSON数据
        data = request.get_json()
        question = data.get('question')
    except Exception as e:
        return jsonify({'error': 'Invalid JSON data'}), 400
        
    # 修复日志记录格式化问题
    log.info("收到问题: %s", question)
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    if vector_store is None:
        return jsonify({'error': 'Vector store not initialized. Please add documents first.'}), 500
    
    # 获取答案
    result = qa_system.ask(question)
    
    # 将结果转换为可JSON序列化的格式
    response_data = {
        "answer": result['answer'],
        "source_documents": [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in result['source_documents']
        ]
    }
    
    return jsonify(response_data)

# 添加新文档到向量存储的接口
@app.route('/llm/add_documents/', methods=['POST'])
def add_documents():
    """
    向现有向量存储中添加新文档
    """
    global vector_store
    
    try:
        # 从请求体中获取JSON数据
        data = request.get_json()
        file_paths = data.get('file_paths', [])
        
        if not file_paths:
            return jsonify({'error': 'file_paths is required'}), 400
        
        # 如果向量存储不存在，创建新的；否则添加到现有向量存储
        if vector_store is None:
            vector_store = processor.process_documents(file_paths)
        else:
            # 添加新文档到现有向量存储
            vector_store = processor.add_documents_to_store(vector_store, file_paths)
        
        # 保存更新后的向量存储
        processor.save_vector_store(vector_store, "src/llm/faiss_index")

        return jsonify({
            "status": "success",
            "message": f"Successfully added documents to vector store"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)