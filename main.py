import argparse
import asyncio
import sys

from app.core import config
from app.services import build_mcp_client, load_readonly_tools
from app.agent import build_agent
from app.api.server import extract_markdown_from_partial_json

# Suporte a UTF-8 no console Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _final_text(result) -> str:
    """Extrai o texto da última mensagem do agente."""
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
                    return call["args"].get("resposta_markdown", "")
                
    if not result.get("messages"):
        return ""
        
    last_msg = result["messages"][-1]
    content = getattr(last_msg, "content", last_msg)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


async def _connect():
    """Conecta ao MCP e retorna (tools, blocked). Levanta exceção em falha."""
    client = build_mcp_client()
    return await load_readonly_tools(client)


async def check_connection() -> int:
    """Testa a conexão com o MCP: valida token, confirma /mcp e lista ferramentas."""
    config.print_config()
    print("\n[CHECK] Conectando ao servidor MCP e listando ferramentas...\n")
    try:
        tools, blocked = await _connect()
    except Exception as e:
        print(f"[FALHOU] Não foi possível conectar/autenticar no MCP:\n  {e}\n")
        print("Verifique: OPENMETADATA_MCP_ENDPOINT, o token (OPENMETADATA_JWT_TOKEN) "
              "e se o app MCP está instalado na instância.")
        return 1

    if not tools:
        print("[ATENÇÃO] Conectou, mas nenhuma ferramenta read-only foi encontrada.")
        print(f"  Ferramentas vistas (bloqueadas/escrita): {blocked or '(nenhuma)'}")
        return 2

    print("[OK] Conexão e autenticação funcionaram.")
    print(f"  Read-only disponíveis ({len(tools)}): {', '.join(t.name for t in tools)}")
    print(f"  Bloqueadas por segurança: {', '.join(blocked) or '(nenhuma)'}")
    return 0


async def main_async():
    parser = argparse.ArgumentParser(
        description="Agente de metadados OpenMetadata para Cartão de TODOS (read-only)."
    )
    parser.add_argument("--query", type=str, help="Executa uma consulta direta e encerra.")
    parser.add_argument("--check", action="store_true",
                        help="Só testa a conexão com o MCP e lista as ferramentas.")
    args = parser.parse_args()

    # Teste de conexão isolado (não carrega o agente)
    if args.check:
        sys.exit(await check_connection())

    # Conecta ao MCP oficial e carrega as ferramentas read-only
    tools, blocked = await _connect()
    if not tools:
        print(
            "[ERRO] Nenhuma ferramenta read-only encontrada no servidor MCP. "
            "Rode 'python main.py --check' para diagnosticar."
        )
        return

    agent = build_agent(tools)

    async def answer(history):
        collected_messages = []

        config_kwargs = {"recursion_limit": config.RECURSION_LIMIT}

        async for event in agent.astream_events({"messages": history}, version="v2", config=config_kwargs):
            kind = event.get("event")
            name = event.get("name")
            
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
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
                        print(text, end="", flush=True)
                        
            elif kind == "on_tool_start":
                print(f"\n🤖 [Orquestrador] Consultando catálogo ({name})...\n", flush=True)
                    
            elif kind == "on_chain_end" and name in ("orquestrador", "supervisor", "LangGraph"):
                output = event["data"].get("output")
                if isinstance(output, dict) and "messages" in output:
                    collected_messages = output["messages"]
                    
        final_text = _final_text({"messages": collected_messages})
        return final_text, collected_messages

    # Execução única
    if args.query:
        query = args.query.strip()
        print(f"Processando consulta: '{query}'...\n")
        print("=" * 60)
        await answer([{"role": "user", "content": query}])
        print("\n" + "=" * 60)
        return

    # Loop interativo (mantém histórico entre turnos)
    print("Olá! Sou o assistente de metadados e linhagem do Cartão de TODOS. 🌟")
    print("Como posso ajudar você hoje com informações sobre tabelas, esquemas, linhagem ou qualidade de dados?")
    print("(Digite 'sair' para encerrar)\n")
    history = []
    while True:
        try:
            user_input = input("Pergunta > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("sair", "exit", "quit"):
                print("Encerrando. Até logo!")
                break

            history.append({"role": "user", "content": user_input})
            print("\n" + "=" * 60)
            text, history = await answer(history)
            print("\n" + "=" * 60 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async())
