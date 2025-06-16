from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
import os
from fastapi.middleware.cors import CORSMiddleware
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import uuid

app = FastAPI(title="DeepSeek Chat API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置 DeepSeek
os.environ["DEEPSEEK_API_KEY"] = "sk-a9e11abbba3f4cfa9cd0bd6987dd2fa3"
llm = ChatDeepSeek(model="deepseek-chat")

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
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
    
    return langchain_messages

@app.post("/chat")
async def chat(request: dict):
    print(request)  # 打印请求
    temp_messages = [
        Message(role="user", content=request["message"])
    ]
    print(temp_messages)
   
    try:
        # 转换消息格式
        messages = convert_messages(temp_messages)
        print(messages, "messages")  # 打印消息
        
        if request.get("stream"):
            # 流式响应
            stream = llm.stream(messages)
            
            async def generate():
                async for chunk in stream:
                    if chunk.content:
                        yield chunk.content
            return generate()
        else:
            # 普通响应
            response = llm.invoke(messages)
            return {
                "response": response.content,
                "model": "deepseek-chat"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 生成唯一文件id
        file_id = str(uuid.uuid4())
        # 获取文件扩展名
        ext = file.filename.split(".")[-1] if "." in file.filename else "txt"
        # 构造保存路径
        save_path = f"uploads/{file_id}.{ext}"
        # 确保 uploads 目录存在
        os.makedirs("uploads", exist_ok=True)
        # 保存文件
        with open(save_path, "wb") as f:
            f.write(await file.read())
        return {"file_id": file_id, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
