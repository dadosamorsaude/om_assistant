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
from langchain_core.messages import HumanMessage
from memory import get_session_history

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
    try:
        from memory import close_pool
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
    allow_origins=["*"],  # Em produção, você pode restringir para as origens do seu app Lovable
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
    global agent_executor
    if not agent_executor:
        yield f"data: {json.dumps({'type': 'error', 'content': 'Agente não inicializado no servidor.'}, ensure_ascii=False)}\n\n"
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
        accumulated_responder_json = ""
        last_streamed_markdown = ""
        is_responding = False

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
    return {"status": "healthy", "agent_loaded": agent_executor is not None}

if __name__ == "__main__":
    import uvicorn
    # Render define a porta dinamicamente via variável de ambiente PORT
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
