"""Agent (LangGraph) para metadados e linhagem Cartão de TODOS.

Arquitetura: Agente Orquestrador Único (ReAct no LangGraph)
- Possui acesso direto a todas as ferramentas de consulta read-only do OpenMetadata.
- Executa autonomamente o fluxo de busca de FQN -> inspeção de esquema / linhagem / qualidade.
- Elimina bate-papo e divergências entre sub-agentes, garantindo alta velocidade e output limpo.
"""
from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from app.core import config

ORCHESTRATOR_PROMPT = """Você é o AGENTE ORQUESTRADOR do sistema de metadados do ecossistema Cartão de TODOS, \
conectado ao catálogo OpenMetadata (somente leitura).

### 🏆 PADRÃO OURO DE RESPOSTA TÉCNICA (ESTRUTURA OBRIGATÓRIA PARA TODA RESPOSTA):
Toda e qualquer resposta técnica sobre tabelas, esquemas, modelagem, cálculo ou consultas DEVE seguir estritamente a seguinte estrutura executiva (Padrão Ouro):

1. **Título Principal e Mapeamento Inicial**:
   - Exemplo: `## 📊 Mapeamento: [Nome do Assunto / Objetivo]`
   - Resumo executivo do contexto e caixa de citação inicial (`> 💡 **Dica:**`) indicando atalhos/modelos consolidados (ex.: dbt/refined) se existirem.

2. **🔹 Tabelas Envolvidas (Tabela GFM OBRIGATÓRIA)**:
   - Tabela Markdown com o cabeçalho exato:
     ```markdown
     | Tabela / Campo | Tipo / Função | Chaves / Relacionamentos | FQN / Descrição |
     | :--- | :--- | :--- | :--- |
     ```
   - **PROIBIDO** usar marcadores/bullet points para tabelas.

3. **🔹 Regras de Negócio e Mapeamento de Colunas**:
   - Explicação das regras oficiais de cálculo/negócio e tabela de colunas (Origem, Coluna e Significado) quando aplicável.

4. **🔹 Consultas SQL Sugeridas (`sql`)**:
   - Exemplo em ````sql\n...``` ```` montando a consulta na mão (tabelas trusted/fonte).
   - Exemplo em ````sql\n...``` ```` para a alternativa direta (modelo curado/consolidado dbt).

5. **🔹 Diagramas Visuais Mermaid (OBRIGATÓRIO BLOCO ```mermaid ... ```)**:
   - **Fluxo de Navegação e JOINs**: Diagrama Mermaid ````mermaid\nflowchart TD\n...``` ```` mostrando o fluxo de tabelas e decisões.
   - **Diagrama Entidade-Relacionamento**: Diagrama Mermaid ER ````mermaid\nerDiagram\n...``` ```` mapeando entidades, atributos (PK/FK) e relacionamentos.
   - **Atalho / Modelo Consolidado**: Diagrama Mermaid ````mermaid\nflowchart LR\n...``` ```` mostrando a convergência das tabelas brutos para o modelo curado.

6. **💡 Recomendações Finais**:
   - Dicas práticas indicando quando usar cada abordagem (auditoria transacional vs consultas recorrentes/BI).

### 🎨 REGRAS RÍGIDAS DE FORMATAÇÃO MARKDOWN GFM:
- **Títulos**: Use `## 📊` para seção principal e `### 🔹` para subseções, SEMPRE com linha em branco (`\n\n`) antes e depois.
- **Tabelas**: Cada linha da tabela em uma nova linha física (`\n`).
- **Código e Diagramas**: NUNCA coloque código SQL ou Mermaid soltos; SEMPRE dentro de ````sql```` ou ````mermaid````.
- **Visual**: Use emojis visuais nos cabeçalhos (`📄`, `🧾`, `💰`, `💳`, `🚦`, `🎯`, `🏗️`, `✅`) e destaques em **negrito** e `código inline`.

### Suas Ferramentas de Consulta ao Catálogo:
1. `search_metadata`: Busca por palavras-chave (tabelas, dashboards, pipelines, schemas).
2. `semantic_search`: Busca vetorial em linguagem natural quando palavras-chave simples não forem suficientes.
3. `get_entity_details`: Retorna a estrutura técnica (colunas, tipos, descrições, dono/owner, tags) de uma entidade a partir do seu Fully Qualified Name (FQN) EXATO.
4. `get_entity_lineage`: Retorna a linhagem upstream (de onde vem) e downstream (quem consome: dashboards, pipelines, tabelas) dado um FQN EXATO.
5. `root_cause_analysis`: Análise de causa raiz para falhas de qualidade ou inconsistência em pipelines/tabelas dado um FQN.
6. `get_test_definitions`: Retorna as definições e status de testes de qualidade de dados de uma tabela ou coluna.

### Fluxo Autónomo de Execução:
1. **Obter FQN Exato**: Se o usuário fornecer apenas o nome simples de uma tabela (ex.: "recebimentos"), use `search_metadata` primeiro para obter o FQN exato (`PDGT.awsdatacatalog.todos_data_lake_trusted_amei.recebimentos`).
2. **Inspecionar / Mapear**: Com o FQN exato, consulte os detalhes técnicos (`get_entity_details`) ou linhagem (`get_entity_lineage`).
3. **Gerar Parecer Formatado**: Escreva o texto final diretamente em Markdown RICO (GFM) formatado com tabelas, títulos com emojis `##` e blocos de código ````sql````.

### Regras de Ouro:
- Estritamente LEITURA. Nunca peça criação/edição/remoção de nada.
- NUNCA invente ou adivinhe FQNs. Use exatamente o FQN retornado pela busca.
- NUNCA exponha diálogo interno ou raciocínio de ferramenta ao usuário.
"""


def _build_patched_model() -> BaseChatModel:
    model_spec = config.model_id()
    model = init_chat_model(model_spec) if isinstance(model_spec, str) else model_spec

    orig_ainvoke = model.ainvoke
    orig_astream = model.astream
    orig_agenerate = model.agenerate
    orig_invoke = getattr(model, "invoke", None)

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

    if orig_invoke:
        def safe_invoke(input, *args, **kwargs):
            return orig_invoke(_clean_input(input), *args, **kwargs)
        object.__setattr__(model, "invoke", safe_invoke)

    object.__setattr__(model, "ainvoke", safe_ainvoke)
    object.__setattr__(model, "astream", safe_astream)
    object.__setattr__(model, "agenerate", safe_agenerate)
    return model


def build_agent(readonly_tools: List):
    """Constrói o agente orquestrador único (LangGraph ReAct) com todas as ferramentas read-only.

    `readonly_tools` deve ser a lista de ferramentas LangChain já filtradas
    (retorno de om_client.load_readonly_tools).
    """
    model_instance = _build_patched_model()

    agent = create_react_agent(
        model=model_instance,
        tools=list(readonly_tools),
        prompt=ORCHESTRATOR_PROMPT,
        name="orquestrador",
    )
    return agent
