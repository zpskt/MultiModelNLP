import os

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import Tongyi
class QASystem:
    def __init__(self, vector_store, model_name="qwen-turbo"):
        """
        初始化问答系统
        :param vector_store: FAISS向量存储
        :param model_name: 使用的模型名称
        """

        # 使用通义千问模型
        self.llm = Tongyi(
            model_name=model_name,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),  # 需要设置API Key
            temperature=0.1,
            max_tokens=512
        )
        
        # 创建检索问答链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
    
    def ask(self, question: str) -> dict:
        """
        提出问题并获取答案
        :param question: 问题文本
        :return: 包含答案和源文档的字典
        """
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }