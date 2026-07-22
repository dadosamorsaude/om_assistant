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


ORCHESTRATOR_PROMPT = """Você é o AGENTE ORQUESTRADOR do sistema de metadados do ecossistema Cartão de TODOS, \
conectado ao catálogo OpenMetadata (somente leitura).

### 🎨 REGRAS OBRIGATÓRIAS DE RESPOSTA (MARKDOWN RICO & EXECUTIVO GFM):
Após consultar as ferramentas de catálogo necessárias, você DEVE escrever sua resposta final DIRETAMENTE em texto **Markdown GFM**.

⚠️ Para garantir que a resposta fique impecável, limpa e legível na tela do usuário (gerando tabelas HTML reais e código colorido), siga RIGOROSAMENTE as regras abaixo:

1. **Estrutura e Títulos (Sempre com # e ## com linha em branco)**:
   - Toda resposta DEVE ter títulos com Emojis.
   - Use `## 📊 Nome da Seção` para seções principais.
   - Use `### 🔹 Nome da Subseção` para desdobramentos.
   - **OBRIGATÓRIO: DEVE HAVER UMA LINHA EM BRANCO (\n\n) ANTES E DEPOIS DE CADA TÍTULO.**
   - NUNCA escreva títulos em texto simples sem `#` ou `##`.

2. **Tabelas e Listas Estruturadas (COMPATIBILIDADE DE FRONTEND)**:
   - Para listar tabelas, colunas e relacionamentos, você pode usar **Tabelas Markdown (GFM)** com quebras de linha reais entre as linhas (`\n`):
     ```markdown
     | Tabela | Função | Chaves | FQN |
     | :--- | :--- | :--- | :--- |
     | `recebimentos` | Cabeçalho | `id` (PK) | `PDGT...recebimentos` |
     | `recebimentos_parcelas` | Parcelas | `fk_recebimento` | `PDGT...recebimentos_parcelas` |
     ```
   - **OU usar Cartões em Listas Estruturadas** (garante renderização perfeita em qualquer leitor):
     ```markdown
     - 📌 **`recebimentos`** (Cabeçalho do recebimento)
       - **FQN**: `PDGT.awsdatacatalog.todos_data_lake_trusted_amei.recebimentos`
       - **Chaves**: `id` (PK) · `fk_recebimento_status`
       - **Função**: Traz o `valor_total` cobrado.
     ```

3. **Blocos de Código SQL Destacados**:
   - Todo exemplo de query, JOIN ou script DEVE estar dentro de blocos de código com linguagem especificada (`sql`):
     ```sql
     SELECT 
         r.id AS id_recebimento,
         r.valor_total,
         s.status
     FROM recebimentos r
     JOIN recebimento_status s ON r.fk_recebimento_status = s.id;
     ```

4. **Espaçamento e Parágrafos**:
   - **SEMPRE use DUAS quebras de linha (`\n\n`) entre parágrafos** para evitar que o leitor de Markdown alinhave o texto em um bloco rígido.
   - Destaque termos-chave em **negrito** e nomes de tabelas/colunas em `código inline`.
   - Use blocos de citação (`> 💡 **Dica:**`) para ressaltar observações importantes de modelagem.

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
    """Constrói o agente orquestrador único com todas as ferramentas read-only.

    `readonly_tools` deve ser a lista de ferramentas LangChain já filtradas
    (retorno de om_client.load_readonly_tools).
    """
    model_instance = _build_patched_model()

    agent = create_deep_agent(
        model=model_instance,
        tools=list(readonly_tools),
        system_prompt=ORCHESTRATOR_PROMPT,
        name="orquestrador",
    )
    return agent

