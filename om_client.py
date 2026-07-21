"""Conexão com o servidor MCP oficial do OpenMetadata.

Substitui o antigo mcp_client.py (SSE/OAuth feito à mão) pelo
`langchain-mcp-adapters`, que reusa a sessão e converte as ferramentas
MCP em ferramentas LangChain automaticamente.

Transporte: SSE em {OPENMETADATA_URL}/mcp
Autenticação: JWT Bearer (Personal Access Token) — com fallback OAuth 2.0.
"""
from typing import List, Tuple

import requests
from langchain_mcp_adapters.client import MultiServerMCPClient

import config

# Allowlist explícito: SOMENTE ferramentas de leitura/análise do MCP oficial.
# Qualquer create_*/patch_* fica de fora por construção (read-only garantido).
# Nomes conforme a referência oficial do OpenMetadata MCP.
READ_ONLY_TOOLS = {
    "search_metadata",      # Discover: busca por palavra-chave
    "semantic_search",      # Discover: busca vetorial em linguagem natural
    "get_entity_details",   # Inspect: detalhes/esquema de uma entidade por FQN
    "get_entity_lineage",   # Lineage: upstream/downstream
    "root_cause_analysis",  # Lineage: análise de causa raiz (somente leitura)
    "get_test_definitions", # Quality: definições de testes de qualidade
}


def _resolve_token() -> str:
    """Obtém o token de autenticação para o header Authorization."""
    if config.OPENMETADATA_JWT_TOKEN:
        return config.OPENMETADATA_JWT_TOKEN

    if config.OPENMETADATA_TOKEN_URL:
        # Fallback OAuth 2.0 Client Credentials
        resp = requests.post(
            config.OPENMETADATA_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": config.OPENMETADATA_CLIENT_ID,
                "client_secret": config.OPENMETADATA_CLIENT_SECRET,
                "scope": "openid email profile",
            },
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if not token:
            raise RuntimeError("OAuth respondeu sem 'access_token'.")
        return token

    raise RuntimeError(
        "Sem credenciais MCP. Defina OPENMETADATA_JWT_TOKEN (recomendado) "
        "ou as variáveis OAuth (OPENMETADATA_TOKEN_URL/CLIENT_SECRET) no .env."
    )


def build_mcp_client() -> MultiServerMCPClient:
    """Cria o cliente MCP apontando para o servidor oficial via HTTP."""
    if not config.OPENMETADATA_MCP_ENDPOINT:
        raise RuntimeError("OPENMETADATA_MCP_ENDPOINT não configurado no .env.")

    token = _resolve_token()
    return MultiServerMCPClient(
        {
            "openmetadata": {
                "transport": "http",
                "url": config.OPENMETADATA_MCP_ENDPOINT,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    )


async def load_readonly_tools(client: MultiServerMCPClient) -> Tuple[List, List[str]]:
    """Carrega as ferramentas do MCP e retorna apenas as read-only.

    Retorna (ferramentas_permitidas, nomes_bloqueados).
    """
    all_tools = await client.get_tools()
    allowed = [t for t in all_tools if t.name in READ_ONLY_TOOLS]
    blocked = sorted({t.name for t in all_tools} - READ_ONLY_TOOLS)
    return allowed, blocked
