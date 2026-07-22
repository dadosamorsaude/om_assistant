"""Deep Agent (LangChain/deepagents) para metadados e linhagem Cartão de TODOS.

Arquitetura: Agente Orquestrador Único (ReAct)
- Possui acesso direto a todas as ferramentas de consulta read-only do OpenMetadata.
- Executa autonomamente o fluxo de busca de FQN -> inspeção de esquema / linhagem / qualidade.
- Elimina bate-papo e divergências entre sub-agentes, garantindo alta velocidade e output limpo.
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
    """Ferramenta obrigatória que o AGENTE DEVE chamar para entregar a resposta final ao usuário.
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


ORCHESTRATOR_PROMPT = """Você é o AGENTE ORQUESTRADOR do sistema de metadados do ecossistema Cartão de TODOS, \
conectado ao catálogo OpenMetadata (somente leitura).

### ⚠️ REGRA CRÍTICA PARA RETORNO DE RESPOSTAS (OBRIGATÓRIO):
Você está PROIBIDO de responder ao usuário final diretamente em texto livre sem invocar a ferramenta.
Toda e qualquer resposta final que você entregar ao usuário DEVE ser enviada chamando a ferramenta `responder_usuario`.
- **Caso a resposta seja TÉCNICA** e envolver dados do catálogo (tabelas, colunas, schemas, linhagem, testes), preencha os parâmetros correspondentes (`ativos_citados`, `colunas`, `linhagem`, `testes_qualidade`) com os dados estruturados obtidos das ferramentas.
- **Caso a resposta seja GERAL** (saudações, agradecimentos ou mensagens sem dados técnicos), chame `responder_usuario` com `intencao="general"`, `resumo` e `resposta_markdown` preenchidos, deixando os outros parâmetros vazios.

### Suas Ferramentas de Consulta ao Catálogo:
Você possui acesso direto a todas as ferramentas read-only do OpenMetadata:
1. `search_metadata`: Busca por palavras-chave (tabelas, dashboards, pipelines, schemas).
2. `semantic_search`: Busca vetorial em linguagem natural quando palavras-chave simples não forem suficientes.
3. `get_entity_details`: Retorna a estrutura técnica (colunas, tipos, descrições, dono/owner, tags) de uma entidade a partir do seu Fully Qualified Name (FQN) EXATO.
4. `get_entity_lineage`: Retorna a linhagem upstream (de onde vem) e downstream (quem consome: dashboards, pipelines, tabelas) dado um FQN EXATO.
5. `root_cause_analysis`: Análise de causa raiz para falhas de qualidade ou inconsistência em pipelines/tabelas dado um FQN.
6. `get_test_definitions`: Retorna as definições e status de testes de qualidade de dados de uma tabela ou coluna.

### Fluxo Autónomo de Execução:
1. **Verificação de FQN**: Se a pergunta do usuário mencionar um termo ou nome simples (ex.: "tabela de recebimentos" ou "recebimentos_zoop") sem o FQN completo (ex.: `PDGT.awsdatacatalog.todos_data_lake_trusted_amei.recebimentos`), use PRIMEIRO `search_metadata` ou `semantic_search` para localizar a entidade e obter o FQN EXATO retornado pelo catálogo.
2. **Inspeção / Linhagem**: De posse do FQN exato, acione imediatamente a ferramenta correspondente ao pedido do usuário (`get_entity_details` para colunas/schema, `get_entity_lineage` para consumo/linhagem, etc.).
3. **Consolidação Final**: Monte um parecer claro, técnico e bem estruturado em Markdown em português e invoque `responder_usuario` com o texto e os dados JSON estruturados.

### Regras de Ouro:
- Estritamente LEITURA. Nunca peça ou tente criação/edição/remoção de dados.
- NUNCA invente ou adivinhe FQNs. Use exatamente a string de FQN retornada pela busca de metadados.
- NUNCA retorne diálogo interno ou raciocínio em texto livre para o usuário. Execute as ferramentas silenciosamente e conclua a resposta chamando a ferramenta `responder_usuario`.
"""


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
    """Constrói o agente orquestrador único com todas as ferramentas read-only e responder_usuario.

    `readonly_tools` deve ser a lista de ferramentas LangChain já filtradas
    (retorno de om_client.load_readonly_tools).
    """
    all_tools = list(readonly_tools) + [responder_usuario]
    model_instance = _build_patched_model()

    agent = create_deep_agent(
        model=model_instance,
        tools=all_tools,
        system_prompt=ORCHESTRATOR_PROMPT,
        name="orquestrador",
    )
    return agent

