# Cartão de TODOS — Deep Agent de Metadados e Linhagem

Sistema **multi-agente** (Supervisor + Especialistas) construído com o framework
**LangChain / [`deepagents`](https://docs.langchain.com/oss/python/deepagents/overview)**
e conectado ao **servidor MCP oficial do OpenMetadata** para responder, em linguagem
natural, sobre tabelas, esquemas, linhagem e qualidade de dados.

O raciocínio e o tool-calling são feitos pela **API da Anthropic (Claude)**.

---

## 🎯 Arquitetura

Padrão **Supervisor + Sub-agentes**, nativo do `deepagents` (que roda sobre LangGraph
e traz planejamento, sub-agentes com contexto isolado, filesystem e streaming prontos):

- **Supervisor** — planeja as etapas (todos) e **delega** via a ferramenta `task`. Não
  acessa o catálogo diretamente; consolida as respostas dos especialistas.
- **discover-agent** — `search_metadata`, `semantic_search` (localiza ativos e FQNs).
- **inspector-agent** — `get_entity_details` (esquema, colunas, dono, tags).
- **lineage-agent** — `get_entity_lineage`, `root_cause_analysis` (consumo/impacto).
- **quality-agent** — `get_test_definitions` (testes de qualidade).

Cada especialista recebe **apenas** as ferramentas da sua função.

## 🛡️ Estritamente Read-Only

A garantia de leitura é feita por **allowlist explícito** em `om_client.py`
(`READ_ONLY_TOOLS`): apenas as 6 ferramentas de leitura/análise do MCP oficial são
carregadas. Toda ferramenta de escrita do OpenMetadata (`create_lineage`,
`create_glossary`, `patch_entity`, `create_test_case`, etc.) é descartada **antes** de
chegar ao agente — o modelo nunca as vê.

## 📁 Estrutura

```
config.py       # variáveis de ambiente e helper de modelo
om_client.py    # conexão MCP (SSE + JWT) e filtro read-only
deep_agent.py   # supervisor + 4 sub-agentes (create_deep_agent)
main.py         # CLI (consulta única ou loop interativo)
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
