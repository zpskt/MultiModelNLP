import os

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

class QASystem:
    def __init__(self, vector_store, model_name="Qwen/Qwen2-1.5B", use_api=False, api_key=None):  # 添加API相关参数
        """
        初始化问答系统
        :param vector_store: FAISS向量存储
        :param model_name: 使用的模型名称
        :param use_api: 是否使用通义模型API
        :param api_key: 通义模型API密钥
        """
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        if use_api:
            # 使用通义模型API的方式
            from langchain_community.llms import Tongyi
            if not api_key:
                raise ValueError("API key is required when using Tongyi API")
            self.llm = Tongyi(
                model_name="qwen-plus",  # 或其他通义模型
                tongyi_api_key=api_key,
                temperature=0.7,
                max_tokens=512
            )
        else:
            # 使用本地pipeline的方式

            # 使用HuggingFacePipeline包装模型以兼容LangChain
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 创建pipeline并包装为HuggingFacePipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)

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