from document_loader import DocumentProcessor
from qa_system import QASystem
import os
from src.util.log import LoggerManager
log = LoggerManager().get_logger()

def main():
    # 初始化文档处理器
    processor = DocumentProcessor()
    os.environ["DASHSCOPE_API_KEY"] = "sk-********"
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 指定文档路径或目录路径（请根据实际情况修改）
    document_paths = [
        # "/Users/zhangpeng/Desktop/计算机科学与工程学院_计算机技术_2201903053_张鹏.docx",
        os.path.join(current_dir, "doc_file", "预约制冰.txt"),
        # "path/to/your/document.xlsx"
        # 或者指定目录路径:
        os.path.join(current_dir, "doc_file")
    ]
    
    # 检查是否有文档路径
    if not document_paths:
        print("请在document_paths列表中添加文档路径或目录路径")
        return
    
    # 处理文档路径，如果是目录则获取其中所有支持的文件
    expanded_document_paths = []
    for path in document_paths:
        if os.path.isfile(path):
            expanded_document_paths.append(path)
        elif os.path.isdir(path):
            try:
                dir_files = processor.get_files_from_directory(path)
                expanded_document_paths.extend(dir_files)
                log.info(f"在目录 {path} 中找到 {len(dir_files)} 个文件")
            except ValueError as e:
                log.error(f"处理目录 {path} 时出错: {e}")
        else:
            log.error(f"路径不存在: {path}")
    
    # 检查是否有有效的文件路径
    if not expanded_document_paths:
        log.error("未找到有效文档文件")
        return
    
    # 处理文档并创建向量存储
    print("Processing documents...")
    vector_store = processor.process_documents(expanded_document_paths)
    
    # 保存向量存储（可选）
    processor.save_vector_store(vector_store, "faiss_index")
    
    # 加载已保存的向量存储（如果之前已保存）
    vector_store = processor.load_vector_store("faiss_index")
    
    # 初始化问答系统

    # gpu服务器使用14B
    # qa_system = QASystem(vector_store, model_name="Qwen/Qwen2-14B ")
    # 本机使用1.5B
    qa_system = QASystem(vector_store, model_name="Qwen/Qwen2-1.5B")

    # 交互式问答
    log.info("文档问答系统已准备就绪！输入 'quit' 退出。")
    while True:
        question = input("\n请输入您的问题: ")
        if question.lower() == 'quit':
            break
        
        try:
            result = qa_system.ask(question)
            log.info(f"\n答案: {result['answer']}")
            
            # 显示参考文档片段
            log.info("\n参考内容:")
            for i, doc in enumerate(result['source_documents']):
                log.info(f"{i+1}. {doc.page_content[:200]}...")
                log.info(f"   来源: {doc.metadata.get('source', 'Unknown')}")
                
        except Exception as e:
            log.error(f"处理问题时出错: {e}")

if __name__ == "__main__":
    main()