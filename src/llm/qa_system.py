from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
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
            return_source_documents=True,
            # 添加prompt模板来控制输出格式
            chain_type_kwargs={
                "prompt": None  # 使用默认prompt，避免添加额外的提示信息
            }
        )
    
    def ask(self, question: str) -> dict:
        """
        提出问题并获取答案
        :param question: 问题文本
        :return: 包含答案和源文档的字典
        """
        # 使用 invoke 方法替代已弃用的 __call__ 方法
        result = self.qa_chain.invoke(question)
        # 提取纯答案，去除可能的提示信息
        answer = result["result"]
        # 移除常见的提示前缀
        if "Answer:" in answer:
            answer = answer.split("Answer:", 1)[1].strip()
        elif "答案:" in answer:
            answer = answer.split("答案:", 1)[1].strip()
        elif "Use the following pieces of context" in answer:
            # 如果模型输出了提示信息，只保留问题相关部分
            lines = answer.split('\n')
            # 找到包含答案的部分
            for i, line in enumerate(lines):
                if line.strip() == '' and i < len(lines) - 1:
                    answer = '\n'.join(lines[i+1:])
                    break
        
        return {
            "answer": answer,
            "source_documents": result["source_documents"]
        }
    
    def _create_custom_prompt(self):
        """
        创建自定义prompt模板来控制输出格式 todo这里加上以后反而效率不好
        """
        from langchain.prompts import PromptTemplate
        
        # 定义自定义prompt模板
        template = """仅基于提供的文档片段回答，不得使用文档外的信息”，从源头避免模型编造内容。
        文档片段: {context}
        问题: {question}
        有用的回答:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        return prompt
