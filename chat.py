import dotenv
dotenv.load_dotenv()
from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Query, Depends, Response
from pydantic import BaseModel
from typing import List, Optional
import os
from fastapi.middleware.cors import CORSMiddleware
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, select, asc, func
import datetime
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader, PDFPlumberLoader, Docx2txtLoader
from langchain.chains import RetrievalQA
from fastapi import File as FastAPIFile
from langchain_community.vectorstores import FAISS
from llm_utils import (
    llm, DashScopeEmbeddings, parse_file, split_docs, ensure_str_docs, build_vector_store
)



app = FastAPI(title="DeepSeek Chat API")

# CORS中间件应在所有路由和依赖之前添加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 OPTIONS 路由兜底，确保所有预检请求都能被正常响应
@app.options("/{rest_of_path:path}")
async def preflight_handler():
    return Response()

# 配置 DeepSeek
os.environ["DEEPSEEK_API_KEY"] = "sk-a9e11abbba3f4cfa9cd0bd6987dd2fa3"
llm = ChatDeepSeek(model="deepseek-chat")

# 数据库配置
DATABASE_URL = "sqlite+aiosqlite:///./chat_history.db"
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()

# 聊天消息表模型
class ChatMessage(Base):
    __tablename__ = "chat_message"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), index=True)
    user_id = Column(String(64), index=True)
    role = Column(String(16))
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# 用户表模型
class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(32), unique=True, index=True)
    password_hash = Column(String(128))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# 密码加密与校验
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def hash_password(password: str):
    return pwd_context.hash(password)
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# JWT 配置
SECRET_KEY = "your_secret_key"  # 请替换为更安全的密钥
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

def create_access_token(data: dict, expires_delta: datetime.timedelta = None):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + (expires_delta or datetime.timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

async def get_current_user(request: Request, token: str = Depends(oauth2_scheme)):
    # 处理CORS预检请求，直接放行
    if request.method == "OPTIONS":
        return None
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()
        if user is None:
            raise credentials_exception
        return user

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    system_prompt: Optional[str] = None

def convert_messages(messages: List[Message], system_prompt: Optional[str] = None):
    langchain_messages = []
    
    # 添加系统提示
    if system_prompt:
        langchain_messages.append(SystemMessage(content=system_prompt))
    
    # 转换消息
    for msg in messages:
        print(msg.role, msg.content)
        if msg.role == "system":
            langchain_messages.append(SystemMessage(content=msg.content))
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
    
    return langchain_messages

# 获取历史消息
async def get_chat_history(session_id, limit=10):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(asc(ChatMessage.timestamp))
        )
        messages = result.scalars().all()
        recent_msgs = messages[-limit:] if len(messages) > limit else messages
        early_msgs = messages[:-limit] if len(messages) > limit else []
        return early_msgs, recent_msgs

# 用大模型自动生成摘要
async def summarize_messages(messages):
    if not messages:
        return ""
    summary_prompt = "请帮我总结以下对话的关键信息：" + " ".join([msg.content for msg in messages])
    # 用大模型生成摘要
    summary = await llm.ainvoke([HumanMessage(content=summary_prompt)])
    print(summary, "summary")
    return summary.content

@app.post("/chat")
async def chat(request: Request, current_user: User = Depends(get_current_user)):
    req = await request.json()
    print(req, '==========')  # 打印请求
    session_id = req.get("session_id")
    if not session_id or not isinstance(session_id, str) or len(session_id) < 8:
        session_id = str(uuid.uuid4())
    user_id = current_user.username
    file_ids = req.get("file_ids")
    try:
        if file_ids:
            # ----------- RAG 多文件问答逻辑 -------------
            all_docs = []
            embeddings = DashScopeEmbeddings( model="text-embedding-v1", dashscope_api_key=os.environ["DASH_SCOPE_API_KEY"])
            async with AsyncSessionLocal() as db:
                for file_id in file_ids:
                    result = await db.execute(select(File).where(File.id == file_id, File.user_id == current_user.username))
                    print(result, "result====")
                    file_obj = result.scalar_one_or_none()
                    if not file_obj:
                        raise HTTPException(status_code=403, detail=f"无权访问文件 {file_id}")
                    persist_path = f"vectorstores/{file_id}"
                    vectorstore = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
                    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
                    docs = retriever.invoke(req["message"])
                    print(docs, "docs====")
                    all_docs.extend(docs)
            unique_docs = {doc.page_content: doc for doc in all_docs}.values()
            context = "\n".join([doc.page_content for doc in unique_docs])
            prompt = f"以下是用户上传的多个文件内容片段：\n{context}\n\n用户问题：{req['message']}\n请基于文件内容作答。"
            if req.get("stream"):
                stream = llm.stream([HumanMessage(content=prompt)])
                async def generate():
                    async for chunk in stream:
                        if chunk.content:
                            yield chunk.content
                return generate()
            else:
                response = llm.invoke([HumanMessage(content=prompt)])
                return {
                    "content": response.content,
                    "sources": [doc.page_content for doc in unique_docs],
                    "model": "deepseek-chat",
                    "session_id": session_id
                }
        else:
            # ----------- 普通对话逻辑 -------------
            early_msgs, recent_msgs = await get_chat_history(session_id, limit=10)
            summary = await summarize_messages(early_msgs)
            context_msgs = []
            if summary:
                context_msgs.append(Message(role="system", content=summary))
            for msg in recent_msgs:
                context_msgs.append(Message(role=msg.role, content=msg.content))
            context_msgs.append(Message(role="user", content=req["message"]))
            messages = convert_messages(context_msgs)
            print(messages, "messages")
            async with AsyncSessionLocal() as db:
                user_msg = ChatMessage(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content=req["message"]
                )
                db.add(user_msg)
                await db.commit()
            if req.get("stream"):
                stream = llm.stream(messages)
                async def generate():
                    async for chunk in stream:
                        if chunk.content:
                            async with AsyncSessionLocal() as db:
                                ai_msg = ChatMessage(
                                    session_id=session_id,
                                    user_id=user_id,
                                    role="assistant",
                                    content=chunk.content
                                )
                                db.add(ai_msg)
                                await db.commit()
                            yield chunk.content
                return generate()
            else:
                response = llm.invoke(messages)
                async with AsyncSessionLocal() as db:
                    ai_msg = ChatMessage(
                        session_id=session_id,
                        user_id=user_id,
                        role="assistant",
                        content=response.content
                    )
                    db.add(ai_msg)
                    await db.commit()
                return {
                    "content": response.content,
                    "model": "deepseek-chat",
                    "session_id": session_id
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
#     try:
#         # 生成唯一文件id
#         file_id = str(uuid.uuid4())
#         # 获取文件扩展名
#         ext = file.filename.split(".")[-1] if "." in file.filename else "txt"
#         # 构造保存路径
#         save_path = f"uploads/{file_id}.{ext}"
#         # 确保 uploads 目录存在
#         os.makedirs("uploads", exist_ok=True)
#         # 保存文件
#         with open(save_path, "wb") as f:
#             f.write(await file.read())
#         return {"file_id": file_id, "filename": file.filename}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "message": "DeepSeek Chat API is running (Langchain Version)",
        "version": "1.0.0",
        "endpoints": {
            "/": "API 信息",
            "/chat": "聊天接口 (POST)",
            "/upload": "文件上传接口 (POST)",
        }
    }

@app.get("/history")
async def get_history(session_id: str = Query(...), limit: int = Query(20), current_user: User = Depends(get_current_user)):
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .where(ChatMessage.user_id == current_user.username)
                .order_by(asc(ChatMessage.timestamp))
                .limit(limit)
            )
            messages = result.scalars().all()
            history = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ]
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_sessions(current_user: User = Depends(get_current_user)):
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(
                    ChatMessage.session_id,
                    func.max(ChatMessage.timestamp).label("last_active"),
                    func.count(ChatMessage.id).label("message_count")
                )
                .where(ChatMessage.user_id == current_user.username)
                .group_by(ChatMessage.session_id)
                .order_by(func.max(ChatMessage.timestamp).desc())
            )
            session_rows = result.fetchall()
            sessions = []
            for row in session_rows:
                # 查询该会话的第一条消息内容作为title
                first_msg_result = await db.execute(
                    select(ChatMessage.content)
                    .where(ChatMessage.session_id == row.session_id)
                    .where(ChatMessage.user_id == current_user.username)
                    .order_by(asc(ChatMessage.timestamp))
                    .limit(1)
                )
                first_msg = first_msg_result.scalar()
                title = first_msg[:10] if first_msg else ""
                sessions.append({
                    "session_id": row.session_id,
                    "last_active": row.last_active.isoformat() if row.last_active else None,
                    "message_count": row.message_count,
                    "title": title
                })
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

@app.post("/register")
async def register(user: UserCreate):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.username == user.username))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="用户名已存在")
        user_obj = User(
            username=user.username,
            password_hash=hash_password(user.password)
        )
        db.add(user_obj)
        await db.commit()
        return {"msg": "注册成功"}

@app.post("/login", response_model=Token)
async def login(user: UserCreate):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.username == user.username))
        user_obj = result.scalar_one_or_none()
        if not user_obj or not verify_password(user.password, user_obj.password_hash):
            raise HTTPException(status_code=400, detail="用户名或密码错误")
        access_token = create_access_token(data={"sub": user.username}, expires_delta=datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        return {"access_token": access_token, "token_type": "bearer"}

# 新增 File 表模型
class File(Base):
    __tablename__ = "file"
    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), index=True)
    filename = Column(String(256))
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)

@app.post("/upload")
async def upload_file_rag(file: UploadFile = FastAPIFile(...), current_user: User = Depends(get_current_user)):
    try:
        file_id = str(uuid.uuid4())
        ext = file.filename.split(".")[-1] if "." in file.filename else "txt"
        save_path = f"uploads/{file_id}.{ext}"
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(await file.read())
        # 解析、分块、embedding、入库
        docs = parse_file(save_path)

        docs = ensure_str_docs(docs)    

        docs = split_docs(docs)

        docs = ensure_str_docs(docs)

        persist_path = f"vectorstores/{file_id}"
        os.makedirs("vectorstores", exist_ok=True)
        build_vector_store(docs, persist_path)
        # 保存文件信息到数据库
        async with AsyncSessionLocal() as db:
            file_obj = File(
                id=file_id,
                user_id=current_user.username,
                filename=file.filename
            )
            db.add(file_obj)
            await db.commit()
        return {"file_id": file_id, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/my_files")
async def my_files(current_user: User = Depends(get_current_user)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(File).where(File.user_id == current_user.username))
        files = result.scalars().all()
        return [{"file_id": f.id, "filename": f.filename, "upload_time": f.upload_time.isoformat()} for f in files]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
