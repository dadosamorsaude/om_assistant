"""Deep Agent (LangChain/deepagents) para metadados e linhagem Cartão de TODOS.

Padrão Supervisor + Especialistas, agora nativo no `deepagents`:
- O agente principal (supervisor) NÃO recebe ferramentas MCP: ele planeja
  (write_todos) e delega via a ferramenta `task` para os sub-agentes.
- Cada sub-agente recebe apenas o subconjunto de ferramentas read-only da
  sua especialidade, mantendo contexto isolado (context quarantine).
"""
from typing import List
import json
from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model

from deepagents import create_deep_agent

import config


@tool
def responder_usuario(
    intencao: str,
    resumo: str,
    resposta_markdown: str,
    ativos_citados: list[dict] = [],
    colunas: list[dict] = [],
    linhagem: dict = {},
    testes_qualidade: list[dict] = []
) -> str:
    """Ferramenta obrigatória que o SUPERVISOR deve chamar para entregar a resposta final ao usuário.
    Toda e qualquer resposta final enviada deve usar esta ferramenta.
    
    Parâmetros:
    - intencao: A intenção identificada ('discover', 'inspect_schema', 'inspect_lineage', 'quality_check' ou 'general').
    - resumo: Resumo rápido da resposta para o usuário final, em texto plano.
    - resposta_markdown: Texto explicativo completo em markdown.
    - ativos_citados: Lista de dicionários para tabelas/ativos encontrados. Cada dicionário DEVE ter as chaves: 'fqn' (Fully Qualified Name exato), 'nome' (nome da tabela), 'tipo' (geralmente 'table'), 'descricao' (descrição da tabela) e 'owner' (dono da tabela).
    - colunas: Lista de colunas de uma tabela (se a intenção for inspect_schema). Cada dicionário DEVE ter as chaves: 'nome' (nome do campo), 'tipo' (tipo de dados) e 'descricao' (descrição).
    - linhagem: Dicionário de linhagem com chaves 'upstream' (lista de FQNs pais) e 'downstream' (lista de FQNs filhos/dashboards que a consomem).
    - testes_qualidade: Lista de testes de qualidade da tabela. Cada dicionário DEVE ter as chaves: 'teste_nome' (nome do teste), 'coluna' (coluna testada) e 'status' (sucesso/falha/etc.).
    """
    return json.dumps({
        "intent": intencao,
        "summary": resumo,
        "detailed_markdown": resposta_markdown,
        "metadata": {
            "entities": ativos_citados,
            "columns": colunas,
            "lineage": linhagem,
            "quality_tests": testes_qualidade
        }
    }, ensure_ascii=False)


def _by_name(tools: List, names: set) -> List:
    return [t for t in tools if t.name in names]


SUPERVISOR_PROMPT = """Você é o SUPERVISOR do sistema de metadados do ecossistema Cartão de TODOS, \
conectado ao catálogo OpenMetadata (somente leitura).

### ⚠️ REGRA CRÍTICA PARA RETORNO DE RESPOSTAS (OBRIGATÓRIO):
Você está PROIBIDO de responder ao usuário final diretamente em texto livre. Toda e qualquer resposta final que você entregar ao usuário DEVE ser enviada chamando a ferramenta `responder_usuario`.
- **Caso a resposta seja TÉCNICA** e envolver dados do catálogo (tabelas, colunas, schemas, linhagem, testes), preencha os parâmetros correspondentes (`ativos_citados`, `colunas`, `linhagem`, `testes_qualidade`) com os dados estruturados obtidos dos especialistas.
- **Caso a resposta seja GERAL** (saudações, agradecimentos ou mensagens sem dados técnicos), chame `responder_usuario` com `intencao="general"`, `resumo` e `resposta_markdown` preenchidos, deixando os outros parâmetros vazios.

Seu papel é analisar a pergunta do usuário, planejar as etapas com a ferramenta de \
planejamento (todos) e DELEGAR cada etapa ao sub-agente especialista correto usando a \
ferramenta `task`. Você não consulta o catálogo diretamente — quem faz isso são os \
especialistas.

Sub-agentes disponíveis:
- discover-agent: encontra tabelas/ativos por palavra-chave ou busca semântica. Use \
  primeiro quando você ainda não tem o FQN exato.
- inspector-agent: retorna esquema, colunas, tipos, dono e tags de uma entidade (por FQN).
- lineage-agent: retorna linhagem upstream/downstream (quem consome / de onde vem) e \
  análise de causa raiz.
- quality-agent: retorna as definições de testes de qualidade de dados.

Fluxo recomendado para perguntas compostas (ex.: "qual a tabela de faturamento e quem \
consome ela?"):
1. Delegue ao discover-agent para localizar a tabela e obter o FQN exato.
2. Com o FQN, delegue em paralelo/sequência ao inspector-agent (campos) e/ou \
   lineage-agent (consumo) conforme o que foi pedido.
3. Consolide as respostas dos especialistas em um parecer final.

Regras:
- Estritamente LEITURA. Nunca peça criação/edição/remoção de nada.
- Ao citar entidades, use sempre o Fully Qualified Name (FQN) exato retornado pela busca; \
  nunca invente ou construa FQNs manualmente.
- Responda em português, claro e estruturado em markdown, citando os FQNs e os \
  consumidores/pipelines/dashboards relevantes.
- Lembra: NUNCA retorne texto direto. Finalize a execução sempre com a chamada da ferramenta `responder_usuario`.
"""

_DISCOVER_PROMPT = """Você é o DISCOVER-AGENT, especialista em descoberta de ativos de dados \
no OpenMetadata. Use `search_metadata` (palavra-chave) e `semantic_search` (linguagem \
natural) para localizar tabelas, dashboards, pipelines e termos. Retorne os resultados \
mais relevantes com o Fully Qualified Name (FQN) EXATO de cada um, tipo de entidade e uma \
breve descrição. Não invente FQNs — reporte apenas o que a ferramenta retornar. Seja \
conciso: entregue a lista de candidatos, não análises longas."""

_INSPECTOR_PROMPT = """Você é o INSPECTOR-AGENT, especialista em inspeção técnica de \
metadados. Use `get_entity_details` passando o entityType e o FQN EXATO recebido do \
supervisor/busca. Reporte: descrição, dono (owner), tags/tier e a lista de colunas com \
tipo e descrição. Não construa FQNs manualmente. Responda em português, com uma tabela \
de colunas quando fizer sentido."""

_LINEAGE_PROMPT = """Você é o LINEAGE-AGENT, especialista em linhagem e análise de impacto. \
Use `get_entity_lineage` para mapear upstream (origens) e downstream (consumidores: \
tabelas, pipelines, dashboards) de uma entidade pelo FQN exato. Use `root_cause_analysis` \
quando o usuário investigar a origem de uma falha de qualidade. Deixe claro quem consome \
o dado e por qual pipeline/processo. Responda em português, de forma objetiva."""

_QUALITY_PROMPT = """Você é o QUALITY-AGENT, especialista em observabilidade e qualidade de \
dados. Use `get_test_definitions` para listar as definições/testes de qualidade \
aplicáveis (nível TABLE ou COLUMN). Reporte os testes relevantes e seus parâmetros. \
Responda em português, de forma objetiva."""


def _build_patched_model() -> BaseChatModel:
    model_spec = config.model_id()
    if not isinstance(model_spec, str):
        return model_spec

    model = init_chat_model(model_spec)

    # Patch original methods to avoid LangChain thinking block missing field validation crash
    orig_ainvoke = model.ainvoke
    orig_astream = model.astream
    orig_agenerate = model.agenerate

    def _clean_message(msg):
        if hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, list):
                new_content = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        if not block.get("thinking"):
                            block["thinking"] = " "
                    elif hasattr(block, "type") and getattr(block, "type") == "thinking":
                        if not getattr(block, "thinking", None):
                            try:
                                block.thinking = " "
                            except Exception:
                                pass
                    new_content.append(block)
                msg.content = new_content
        return msg

    def _clean_input(input_data):
        if isinstance(input_data, list):
            return [_clean_message(m) for m in input_data]
        elif isinstance(input_data, dict) and "messages" in input_data:
            input_data["messages"] = [_clean_message(m) for m in input_data["messages"]]
        return input_data

    async def safe_ainvoke(input, *args, **kwargs):
        return await orig_ainvoke(_clean_input(input), *args, **kwargs)

    async def safe_astream(input, *args, **kwargs):
        return await orig_astream(_clean_input(input), *args, **kwargs)

    async def safe_agenerate(messages, *args, **kwargs):
        cleaned_messages = [[_clean_message(m) for m in msg_list] for msg_list in messages]
        return await orig_agenerate(cleaned_messages, *args, **kwargs)

    object.__setattr__(model, "ainvoke", safe_ainvoke)
    object.__setattr__(model, "astream", safe_astream)
    object.__setattr__(model, "agenerate", safe_agenerate)
    return model


def build_agent(readonly_tools: List):
    """Constrói o deep agent supervisor com os 4 sub-agentes especialistas.

    `readonly_tools` deve ser a lista de ferramentas LangChain já filtradas
    (retorno de om_client.load_readonly_tools).
    """
    subagents = [
        {
            "name": "discover-agent",
            "description": (
                "Encontra tabelas, dashboards, pipelines e termos por palavra-chave ou "
                "busca semântica. Use PRIMEIRO quando não souber o FQN exato."
            ),
            "system_prompt": _DISCOVER_PROMPT,
            "tools": _by_name(readonly_tools, {"search_metadata", "semantic_search"}),
        },
        {
            "name": "inspector-agent",
            "description": (
                "Retorna esquema técnico de uma entidade (colunas, tipos, dono, tags) a "
                "partir do FQN exato. Use quando o usuário quer os campos de uma tabela."
            ),
            "system_prompt": _INSPECTOR_PROMPT,
            "tools": _by_name(readonly_tools, {"get_entity_details"}),
        },
        {
            "name": "lineage-agent",
            "description": (
                "Retorna linhagem upstream/downstream (quem consome, de onde vem) e "
                "análise de causa raiz. Use para perguntas de consumo/impacto."
            ),
            "system_prompt": _LINEAGE_PROMPT,
            "tools": _by_name(readonly_tools, {"get_entity_lineage", "root_cause_analysis"}),
        },
        {
            "name": "quality-agent",
            "description": (
                "Lista definições de testes de qualidade de dados (TABLE/COLUMN). Use "
                "para perguntas sobre testes/validação de uma tabela."
            ),
            "system_prompt": _QUALITY_PROMPT,
            "tools": _by_name(readonly_tools, {"get_test_definitions"}),
        },
    ]

    model_instance = _build_patched_model()

    # O supervisor recebe a ferramenta responder_usuario para encapsular o output
    agent = create_deep_agent(
        model=model_instance,
        tools=[responder_usuario],
        system_prompt=SUPERVISOR_PROMPT,
        subagents=subagents,
        name="supervisor",
    )
    return agent
