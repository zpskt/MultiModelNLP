import os

from langchain.chains import RetrievalQA


class QASystem:
    def __init__(self, vector_store, model_name="gpt-3.5-turbo"):
        """
        初始化问答系统
        :param vector_store: FAISS向量存储
        :param model_name: 使用的模型名称
        """

        # 默认使用HuggingFace模型（需要安装transformers）
        from langchain.llms import HuggingFaceHub
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.1, "max_length": 512}
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