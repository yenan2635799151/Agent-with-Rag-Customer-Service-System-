from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import Request
from pydantic import BaseModel
from agent.react_agent import ReactAgent
import json
import time
from utils.ip_context import request_ip
app = FastAPI(title="智能扫地机器人客服 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ReactAgent()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    response: str


def generate_stream(query: str):
    try:
        for chunk in agent.execute_stream(query):
            for char in chunk :
                time.sleep(0.01)
                yield f"data: {json.dumps({'content': char})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_msg = str(e)
        if "Insufficient Balance" in error_msg:
            content = "\n\n抱歉，当前API密钥余额不足，请联系管理员充值。"
            yield f"data: {json.dumps({'content': content})}\n\n"
        else:
            content = f"\n\n发生错误：{error_msg}"
            yield f"data: {json.dumps({'content': content})}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(request:Request,body: ChatRequest):

    xff = request.headers.get("X-Forwarded-For")
    if xff:
        ip = xff.split(",")[0].strip()
    else:
        ip = request.client.host
    request_ip.set(ip)

    return StreamingResponse(
        generate_stream(body.message),
        media_type="text/event-stream",
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
