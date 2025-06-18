import os
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
import dotenv
dotenv.load_dotenv()
# LLM 初始化
os.environ["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "sk-a9e11abbba3f4cfa9cd0bd6987dd2fa3")
llm = ChatDeepSeek(model="deepseek-chat")

# Embedding 初始化
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", os.environ["DASH_SCOPE_API_KEY"])
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 文件解析

def parse_file(file_path):
    if file_path.endswith('.pdf'):
        loader = PDFPlumberLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    print(docs, "docs")
    for doc in docs:
        if hasattr(doc, 'page_content') and isinstance(doc.page_content, bytes):
            doc.page_content = doc.page_content.decode('utf-8', errors='ignore')
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def ensure_str_docs(docs):
    for doc in docs:
        if hasattr(doc, 'page_content') and isinstance(doc.page_content, bytes):
            doc.page_content = doc.page_content.decode('utf-8', errors='ignore')
        elif hasattr(doc, 'page_content') and not isinstance(doc.page_content, str):
            doc.page_content = str(doc.page_content)
    return docs

def build_vector_store(docs, persist_path):
    print(os.environ["DASHSCOPE_API_KEY"], "DASHSCOPE_API_KEY===============================")
    embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=os.environ["DASHSCOPE_API_KEY"])
    print(embeddings, "embeddings")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_path)
    print(vectorstore, "vectorstore")
    return vectorstore 