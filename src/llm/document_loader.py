import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
# 添加更多文档加载器
from langchain_community.document_loaders import TextLoader, UnstructuredPowerPointLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-large")


    def load_document(self, file_path: str) -> List:
        """根据文件类型加载文档"""
        _, ext = os.path.splitext(file_path)

        if ext.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext.lower() == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext.lower() in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path)
        # 添加更多文件类型支持
        elif ext.lower() == '.txt':
            loader = TextLoader(file_path)
        elif ext.lower() in ['.pptx', '.ppt']:
            loader = UnstructuredPowerPointLoader(file_path)
        elif ext.lower() == '.html':
            loader = UnstructuredHTMLLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return loader.load()

    def process_documents(self, file_paths: List[str]) -> FAISS:
        """处理多个文档并创建向量存储"""
        all_documents = []

        for file_path in file_paths:
            if os.path.exists(file_path):
                documents = self.load_document(file_path)
                all_documents.extend(documents)
                print(f"Loaded {len(documents)} documents from {file_path}")
            else:
                print(f"File not found: {file_path}")

        # 分割文档
        split_documents = self.text_splitter.split_documents(all_documents)
        print(f"Split into {len(split_documents)} chunks")
        try:
            # 创建向量存储
            vector_store = FAISS.from_documents(split_documents, self.embeddings)
            return vector_store
        except ValueError as e:
            print(f"Error creating vector store: {e}")
            # 可以在这里添加备选方案
            raise e

    def  add_documents_to_store(self, vector_store: FAISS, file_paths: List[str]) -> FAISS:
        """向现有向量存储中添加新文档"""
        all_documents = []

        for file_path in file_paths:
            if os.path.exists(file_path):
                documents = self.load_document(file_path)
                all_documents.extend(documents)
                print(f"Loaded {len(documents)} documents from {file_path}")
            else:
                print(f"File not found: {file_path}")

        # 分割文档
        split_documents = self.text_splitter.split_documents(all_documents)
        print(f"Split into {len(split_documents)} chunks")

        # 添加到现有向量存储
        vector_store.add_documents(split_documents)
        print(f"Added {len(split_documents)} new chunks to vector store")
        return vector_store

    def get_files_from_directory(self, directory_path: str) -> List[str]:
        """
        从目录中获取所有支持的文件路径
        :param directory_path: 目录路径
        :return: 支持的文件路径列表
        """
        supported_extensions = {'.pdf', '.docx', '.xlsx', '.xls', '.pptx', '.ppt', '.txt', '.md', '.csv', '.tsv', '.json', '.jsonl', '.yaml', '.yml','.doc', '.html'}
        file_paths = []
        
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
            
        if not os.path.isdir(directory_path):
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in supported_extensions:
                    file_paths.append(os.path.join(root, file))
        
        return file_paths

    def save_vector_store(self, vector_store: FAISS, path: str):
        """保存向量存储到磁盘"""
        vector_store.save_local(path)
        print(f"Vector store saved to {path}")

    def load_vector_store(self, path: str) -> FAISS:
        """从磁盘加载向量存储"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store path does not exist: {path}")
        
        # 检查FAISS索引文件是否存在
        index_file = os.path.join(path, "index.faiss")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
            
        try:
            return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            raise Exception(f"Failed to load vector store from {path}: {str(e)}")