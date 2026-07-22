import os
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env
load_dotenv()

# --- Anthropic / LLM ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Modelo Anthropic. Pode ser atualizado livremente (ex: claude-sonnet-4-5).
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")

# --- OpenMetadata MCP ---
OPENMETADATA_URL = os.getenv("OPENMETADATA_URL")
# Endpoint SSE do MCP oficial. Normalmente {OPENMETADATA_URL}/mcp
OPENMETADATA_MCP_ENDPOINT = os.getenv("OPENMETADATA_MCP_ENDPOINT")

# Autenticação recomendada pelo OpenMetadata: JWT / Personal Access Token (PAT).
# Gere em: {OPENMETADATA_URL}/users/<usuario>/access-token
OPENMETADATA_JWT_TOKEN = os.getenv("OPENMETADATA_JWT_TOKEN")

# Fallback: OAuth 2.0 Client Credentials (usado apenas se não houver JWT direto).
OPENMETADATA_CLIENT_ID = os.getenv("OPENMETADATA_CLIENT_ID", "OpenMetadata")
OPENMETADATA_CLIENT_SECRET = os.getenv("OPENMETADATA_CLIENT_SECRET")
OPENMETADATA_TOKEN_URL = os.getenv("OPENMETADATA_TOKEN_URL")
OPENMETADATA_AUTH_PROVIDER = os.getenv("OPENMETADATA_AUTH_PROVIDER", "openmetadata")

# --- Execution / Graph Settings ---
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "50"))

# --- Security / API Authentication ---
API_KEY = os.getenv("API_KEY")

# --- Persistent DB Connection (PostgreSQL) ---
DATABASE_URL = os.getenv("DATABASE_URL")


def model_id() -> str:
    """String de modelo no formato provider:model exigido pelo LangChain/deepagents."""
    return f"anthropic:{ANTHROPIC_MODEL}"


def print_config():
    print("=" * 60)
    print("   CONFIGURAÇÃO DO DEEP AGENT — METADADOS CARTÃO DE TODOS")
    print("=" * 60)
    print(f"Modelo (LangChain):      {model_id()}")
    print(f"Recursion Limit:         {RECURSION_LIMIT}")
    print(f"OpenMetadata URL:        {OPENMETADATA_URL}")
    print(f"OpenMetadata MCP (SSE):  {OPENMETADATA_MCP_ENDPOINT}")
    auth = "JWT/PAT direto" if OPENMETADATA_JWT_TOKEN else f"OAuth 2.0 ({OPENMETADATA_AUTH_PROVIDER})"
    print(f"Autenticação MCP:        {auth}")
    print("=" * 60)

