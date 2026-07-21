import sys

# Suporte a UTF-8 no console Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
import json
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from contextlib import asynccontextmanager

import config
from om_client import build_mcp_client, load_readonly_tools
from deep_agent import build_agent
from main import _final_text

# --- Autenticação via API Key ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key do servidor não configurada nas variáveis de ambiente (API_KEY)."
        )
    if api_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credencial inválida."
        )
    return api_key

# Carregar agente globalmente para reuso entre chamadas
agent_executor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    try:
        print("🤖 Inicializando conexão com MCP e carregando agente...")
        client = build_mcp_client()
        tools, _ = await load_readonly_tools(client)
        agent_executor = build_agent(tools)
        print("🤖 Agente carregado e pronto para receber requisições!")
    except Exception as e:
        print(f"❌ Erro crítico ao inicializar agente: {e}")
    yield
    print("🤖 Encerrando servidor da API...")

app = FastAPI(
    title="API de Metadados e Linhagem — Cartão de TODOS",
    lifespan=lifespan
)

# Configuração de CORS para permitir requisições do Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, você pode restringir para as origens do seu app Lovable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

async def stream_generator(message: str):
    global agent_executor
    if not agent_executor:
        yield f"data: {json.dumps({'type': 'error', 'content': 'Agente não inicializado no servidor.'}, ensure_ascii=False)}\n\n"
        return
        
    try:
        async for event in agent_executor.astream_events({"messages": [{"role": "user", "content": message}]}, version="v2"):
            kind = event.get("event")
            name = event.get("name")
            
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    text = ""
                    if isinstance(chunk.content, str):
                        text = chunk.content
                    elif isinstance(chunk.content, list):
                        for block in chunk.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text += block.get("text", "")
                            elif isinstance(block, str):
                                text += block
                    if text:
                        yield f"data: {json.dumps({'type': 'token', 'content': text}, ensure_ascii=False)}\n\n"
                        
            elif kind == "on_tool_start":
                if name == "task":
                    subagent = event["data"].get("input", {}).get("subagent_type", "especialista")
                    desc = event["data"].get("input", {}).get("description", "")
                    yield f"data: {json.dumps({'type': 'step', 'content': f'Acionando especialista: {subagent} ({desc})'}, ensure_ascii=False)}\n\n"
                elif name == "responder_usuario":
                    yield f"data: {json.dumps({'type': 'step', 'content': 'Formatando resposta final...'}, ensure_ascii=False)}\n\n"
                    
            elif kind == "on_tool_end":
                if name == "responder_usuario":
                    output = event["data"].get("output")
                    if output:
                        val = output
                        if hasattr(val, "content"):
                            val = val.content
                        if isinstance(val, str):
                            try:
                                parsed = json.loads(val)
                                yield f"data: {json.dumps({'type': 'final', 'data': parsed}, ensure_ascii=False)}\n\n"
                            except Exception as e:
                                print(f"Error parsing final JSON: {e}")
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

@app.post("/api/chat", dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    return StreamingResponse(stream_generator(request.message), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "healthy", "agent_loaded": agent_executor is not None}

if __name__ == "__main__":
    import uvicorn
    # Render define a porta dinamicamente via variável de ambiente PORT
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
