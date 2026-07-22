# Cartão de TODOS — Agente de Metadados e Linhagem (LangGraph)

Sistema **Agente Orquestrador Único (ReAct)** construído com o framework
**LangChain / [`LangGraph`](https://langchain-ai.github.io/langgraph/)**
e conectado ao **servidor MCP oficial do OpenMetadata** para responder, em linguagem
natural, sobre tabelas, esquemas, linhagem e qualidade de dados.

O raciocínio e o tool-calling são feitos pelo modelo de linguagem configurado (ex.: Anthropic Claude / Gemini).

---

## 🎯 Arquitetura

Padrão **Agente Orquestrador Único (ReAct no LangGraph)**:
- **Orquestrador** — Possui acesso direto a todas as 6 ferramentas de consulta read-only do OpenMetadata.
- Executa autonomamente o fluxo de busca de FQN -> inspeção de esquema / linhagem / qualidade.
- Gera respostas no **Padrão Ouro** formatadas em Markdown rico com tabelas, consultas SQL e diagramas Mermaid.

---

## 🛡️ Estritamente Read-Only

A garantia de leitura é feita por **allowlist explícito** em `om_client.py`
(`READ_ONLY_TOOLS`): apenas as 6 ferramentas de leitura/análise do MCP oficial são
carregadas. Toda ferramenta de escrita do OpenMetadata (`create_lineage`,
`create_glossary`, `patch_entity`, `create_test_case`, etc.) é descartada **antes** de
chegar ao agente — o modelo nunca as vê.

## 📁 Estrutura

```
app/
├── core/
│   └── config.py        # Configurações e variáveis de ambiente
├── services/
│   ├── om_client.py     # Conexão MCP (SSE + JWT) e filtro read-only
│   └── memory.py        # Histórico de mensagens e sessão (PostgreSQL/InMemory)
├── agent/
│   └── agent.py         # Agente orquestrador único (langgraph.prebuilt.create_react_agent)
└── api/
    └── server.py        # Servidor FastAPI com streaming SSE e rotas da API
api.py                   # Ponto de entrada do servidor (atalho para app.api.server)
main.py                  # CLI (consulta única ou loop interativo)
```

## ⚙️ Configuração

Copie o exemplo e preencha:

```bash
cp .env.example .env
```

Parâmetros principais:
- `ANTHROPIC_API_KEY` — chave da Anthropic.
- `ANTHROPIC_MODEL` — modelo Claude (ex.: `claude-sonnet-4-5`).
- `OPENMETADATA_MCP_ENDPOINT` — endpoint SSE do MCP, normalmente `{URL}/mcp`.
- `OPENMETADATA_JWT_TOKEN` — **recomendado**. Personal Access Token gerado em
  `{OPENMETADATA_URL}/users/<seu-usuario>/access-token`.
- Alternativa: as variáveis `OPENMETADATA_CLIENT_*` / `OPENMETADATA_TOKEN_URL` para
  OAuth 2.0 Client Credentials (usado apenas se o JWT estiver vazio).

## 🚀 Instalar e executar (uv)

O projeto usa o [**uv**](https://docs.astral.sh/uv/) como gerenciador de pacotes.
As dependências ficam no `pyproject.toml` e o `.python-version` fixa o Python **3.12**
(o uv baixa automaticamente se não existir; `deepagents` exige 3.11+).

Instalar o uv (uma vez):

```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Criar o ambiente e instalar tudo:

```bash
uv sync
```

Testar a conexão com o MCP (valida token e lista as ferramentas, sem carregar o agente):

```bash
uv run python main.py --check
```

Consulta única:

```bash
uv run python main.py --query "Qual a tabela de faturamento e quem consome ela?"
```

Modo interativo (mantém histórico entre perguntas):

```bash
uv run python main.py
```

## 🔎 Observabilidade (opcional)

Como roda sobre LangGraph, basta habilitar o LangSmith via ambiente para rastrear as
delegações e chamadas de ferramenta:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
```
