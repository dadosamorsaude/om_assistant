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

from app.core import config
from app.services.om_client import build_mcp_client, load_readonly_tools
from app.services.memory import get_session_history, close_pool
from app.agent.agent import build_agent
from langchain_core.messages import HumanMessage

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
agent_initialization_error = None

async def _init_agent():
    global agent_executor, agent_initialization_error
    try:
        print("🤖 Inicializando conexão com MCP e carregando agente...")
        client = build_mcp_client()
        tools, _ = await load_readonly_tools(client)
        agent_executor = build_agent(tools)
        agent_initialization_error = None
        print("🤖 Agente carregado e pronto para receber requisições!")
        return agent_executor
    except Exception as e:
        agent_initialization_error = str(e)
        print(f"❌ Erro crítico ao inicializar agente: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    await _init_agent()
    yield
    print("🤖 Encerrando servidor da API...")
    try:
        await close_pool()
        print("🤖 Pool de conexões do PostgreSQL encerrado com sucesso.")
    except Exception as e:
        print(f"⚠️ Erro ao fechar pool de conexões do PostgreSQL: {e}")

app = FastAPI(
    title="API de Metadados e Linhagem — Cartão de TODOS",
    lifespan=lifespan
)

# Configuração de CORS para permitir requisições do Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str | None = None

def extract_markdown_from_partial_json(json_str: str) -> str:
    """Extrai incrementalmente o valor do campo 'resposta_markdown' a partir de uma string JSON parcial."""
    idx = json_str.find('"resposta_markdown"')
    if idx == -1:
        idx = json_str.find("'resposta_markdown'")
    if idx == -1:
        return ""
    
    colon_idx = json_str.find(":", idx)
    if colon_idx == -1:
        return ""
    
    quote_idx = json_str.find('"', colon_idx)
    if quote_idx == -1:
        return ""
    
    start_pos = quote_idx + 1
    res = []
    i = start_pos
    n = len(json_str)
    while i < n:
        c = json_str[i]
        if c == '\\':
            if i + 1 < n:
                nxt = json_str[i+1]
                if nxt == 'n':
                    res.append('\n')
                elif nxt == 't':
                    res.append('\t')
                elif nxt == 'r':
                    res.append('\r')
                elif nxt == '"':
                    res.append('"')
                elif nxt == '\\':
                    res.append('\\')
                else:
                    res.append(nxt)
                i += 2
                continue
            else:
                break
        elif c == '"':
            break
        else:
            res.append(c)
            i += 1
            
    return "".join(res)


async def stream_generator(message: str, session_id: str):
    global agent_executor, agent_initialization_error
    if not agent_executor:
        agent_executor = await _init_agent()

    if not agent_executor:
        err_msg = f"Agente não inicializado no servidor. Erro de inicialização: {agent_initialization_error or 'Conexão MCP ou credenciais pendentes.'}"
        yield f"data: {json.dumps({'type': 'error', 'content': err_msg}, ensure_ascii=False)}\n\n"
        return

    try:
        # Recupera histórico persistido do PostgreSQL ou in-memory
        history = get_session_history(session_id)
        
        # Limita o histórico de contexto às últimas 10 mensagens
        recent_messages = list(history.messages)[-10:]
        
        # Remove atributos 'name' das mensagens do histórico para compatibilidade
        for m in recent_messages:
            if hasattr(m, 'name'):
                m.name = None
            m.additional_kwargs.pop('name', None)
            
        input_messages = recent_messages + [HumanMessage(content=message)]
        
        # Salva a mensagem do usuário no histórico persistido
        history.add_user_message(message)
        
        final_text = ""

        config_kwargs = {"recursion_limit": config.RECURSION_LIMIT}

        async for event in agent_executor.astream_events({"messages": input_messages}, version="v2", config=config_kwargs):
            kind = event.get("event")
            name = event.get("name")
            
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                
                # Transmite tokens de conteúdo direto da LLM (Markdown limpo)
                if hasattr(chunk, "content") and chunk.content:
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
                        final_text += text
                        yield f"data: {json.dumps({'type': 'token', 'content': text}, ensure_ascii=False)}\n\n"
                        
            elif kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'step', 'content': f'Consultando catálogo ({name})...'}, ensure_ascii=False)}\n\n"
                                
        # Salva a resposta final do assistente no histórico persistido
        if final_text:
            history.add_ai_message(final_text)
            
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

@app.post("/api/chat", dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    sess_id = request.session_id or request.user_id or "default_session"
    return StreamingResponse(stream_generator(request.message, sess_id), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {
        "status": "healthy" if agent_executor is not None else "degraded",
        "agent_loaded": agent_executor is not None,
        "error": agent_initialization_error,
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=port, reload=False)
