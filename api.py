import sys

# Suporte a UTF-8 no console Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
from fastapi import FastAPI, HTTPException, Security, Depends, status
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

@app.post("/api/chat", dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    global agent_executor
    if not agent_executor:
        raise HTTPException(
            status_code=500, 
            detail="O agente não pôde ser inicializado. Verifique os logs do servidor."
        )
    
    try:
        # Roda o agente com o input do usuário
        result = await agent_executor.ainvoke({"messages": [{"role": "user", "content": request.message}]})
        
        # Procura se o supervisor chamou a ferramenta responder_usuario (de trás para frente)
        structured_response = None
        for msg in reversed(result["messages"]):
            is_user = False
            if isinstance(msg, dict):
                is_user = msg.get("role") == "user"
            else:
                is_user = getattr(msg, "role", None) == "user" or getattr(msg, "type", None) == "human" or msg.__class__.__name__ == "HumanMessage"
            
            if is_user:
                break
                
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    if call["name"] == "responder_usuario":
                        structured_response = call["args"]
                        break
            if structured_response:
                break
        
        if structured_response:
            return {
                "intent": structured_response.get("intencao", "general"),
                "summary": structured_response.get("resumo", ""),
                "detailed_markdown": structured_response.get("resposta_markdown", ""),
                "metadata": {
                    "entities": structured_response.get("ativos_citados", []),
                    "columns": structured_response.get("colunas", []),
                    "lineage": structured_response.get("linhagem", {}),
                    "quality_tests": structured_response.get("testes_qualidade", [])
                }
            }
        
        # Fallback caso seja conversa geral ou texto plano
        response_text = _final_text(result)
        return {
            "intent": "general",
            "summary": response_text[:100] + "..." if len(response_text) > 100 else response_text,
            "detailed_markdown": response_text,
            "metadata": {
                "entities": [],
                "columns": [],
                "lineage": {},
                "quality_tests": []
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "agent_loaded": agent_executor is not None}

if __name__ == "__main__":
    import uvicorn
    # Render define a porta dinamicamente via variável de ambiente PORT
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
