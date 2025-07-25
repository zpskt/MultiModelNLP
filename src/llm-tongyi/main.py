from document_loader import DocumentProcessor
from qa_system import QASystem

def main():
    # 初始化文档处理器
    processor = DocumentProcessor()
    import os
    os.environ["DASHSCOPE_API_KEY"] = "sk-********"
    # 指定文档路径（请根据实际情况修改）
    document_paths = [
        "/Users/zhangpeng/Desktop/计算机科学与工程学院_计算机技术_2201903053_张鹏.docx",
        # "path/to/your/document.docx",
        # "path/to/your/document.xlsx"
    ]
    
    # 检查是否有文档路径
    if not document_paths:
        print("请在document_paths列表中添加文档路径")
        # 演示直接使用LLM功能
        demo_llm_usage()
        return
    
    # 处理文档并创建向量存储
    print("Processing documents...")
    vector_store = processor.process_documents(document_paths)
    
    # 保存向量存储（可选）
    processor.save_vector_store(vector_store, "faiss_index")
    
    # 加载已保存的向量存储（如果之前已保存）
    vector_store = processor.load_vector_store("faiss_index")
    
    # 初始化问答系统

    # 如果不使用OpenAI，则使用免费的HuggingFace模型
    qa_system = QASystem(vector_store, model_name="qwen-turbo")  # 或 "qwen-plus", "qwen-max"
    
    # 交互式问答
    print("文档问答系统已准备就绪！输入 'quit' 退出。")
    while True:
        question = input("\n请输入您的问题: ")
        if question.lower() == 'quit':
            break
        
        try:
            result = qa_system.ask(question)
            print(f"\n答案: {result['answer']}")
            
            # 显示参考文档片段
            print("\n参考内容:")
            for i, doc in enumerate(result['source_documents']):
                print(f"{i+1}. {doc.page_content[:200]}...")
                print(f"   来源: {doc.metadata.get('source', 'Unknown')}")
                
        except Exception as e:
            print(f"处理问题时出错: {e}")

def demo_llm_usage():
    """
    演示直接使用LLM进行对话
    """
    print("\n=== LLM直接对话演示 ===")
    print("1. 使用OpenAI (需要API密钥)")
    print("2. 使用HuggingFace免费模型")
    
    choice = input("请选择 (1 或 2): ")
    
    if choice == "1":
        api_key = input("请输入OpenAI API密钥: ")
        if not api_key:
            print("未提供API密钥，使用默认模型")
            llm_demo_with_hf()
        else:
            llm_demo_with_openai(api_key)
    else:
        llm_demo_with_hf()

def llm_demo_with_openai(api_key):
    """
    使用OpenAI模型进行对话
    """
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
    
    print("\nOpenAI模型已初始化，输入 'quit' 退出。")
    while True:
        question = input("\n请输入您的问题: ")
        if question.lower() == 'quit':
            break
            
        try:
            response = llm.predict(question)
            print(f"\n回答: {response}")
        except Exception as e:
            print(f"处理问题时出错: {e}")

def llm_demo_with_hf():
    """
    使用HuggingFace免费模型进行对话
    """
    try:
        from langchain.llms import HuggingFaceHub
        
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.1, "max_length": 512}
        )
        
        print("\nHuggingFace模型已初始化，输入 'quit' 退出。")
        while True:
            question = input("\n请输入您的问题: ")
            if question.lower() == 'quit':
                break
                
            try:
                response = llm.predict(question)
                print(f"\n回答: {response}")
            except Exception as e:
                print(f"处理问题时出错: {e}")
    except Exception as e:
        print(f"HuggingFace模型初始化失败: {e}")
        print("请确保已安装相关依赖")

if __name__ == "__main__":
    main()