import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
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
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def load_document(self, file_path: str) -> List:
        """根据文件类型加载文档"""
        _, ext = os.path.splitext(file_path)

        if ext.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext.lower() == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext.lower() in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path)
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

    def save_vector_store(self, vector_store: FAISS, path: str):
        """保存向量存储到磁盘"""
        vector_store.save_local(path)
        print(f"Vector store saved to {path}")

    def load_vector_store(self, path: str) -> FAISS:
        """从磁盘加载向量存储"""
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
