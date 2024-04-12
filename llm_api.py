import os
from dotenv import load_dotenv
from typing import List, Dict
from openai import AsyncOpenAI
from datasets import load_dataset
from haystack.agents.base import Tool
from haystack.agents.conversational import ConversationalAgent
from haystack.agents.memory import ConversationSummaryMemory
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, PromptNode
from haystack.pipelines import DocumentSearchPipeline
import chainlit as cl
''
# Carregar variáveis de ambiente
_ = load_dotenv('.env')

# Chave da API da OpenAI
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Por favor, defina a variável de ambiente OPENAI_API_KEY")

# Configurações do modelo OpenAI
config_modelo = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 256,
    "stop_words": ["Observação:"],
}

# Função para interagir com o chatbot da OpenAI de forma assíncrona
async def openai_chatbot_chain(mensagens: List[Dict[str, str]], settings: dict = config_modelo):
    cliente = AsyncOpenAI(api_key=openai_api_key)
    resposta_stream = await cliente.chat.completions.create(
        messages=mensagens, stream=True, **settings
    )
    return resposta_stream

# Função para obter o recuperador de documentos
@cl.cache
def obter_recuperador():
    armazenamento_documentos = InMemoryDocumentStore(use_bm25=True)
    conjunto_dados = load_dataset("bilgeyucel/seven-wonders", split="train")
    armazenamento_documentos.write_documents(conjunto_dados)
    return BM25Retriever(armazenamento_documentos)

# Função para obter o agente conversacional
@cl.cache
def obter_agente(recuperador):
    pipeline = DocumentSearchPipeline(recuperador)
    ferramenta_pesquisa = Tool(
        name="seven_wonders_search",
        pipeline_or_node=pipeline,
        description="útil quando você precisa responder perguntas sobre as sete maravilhas do mundo: Colosso de Rodes, Estátua de Zeus, Grande Pirâmide de Gizé, Mausoléu de Halicarnasso, Templo de Ártemis, Farol de Alexandria e Jardins Suspensos da Babilônia",
        output_variable="documents",
    )
    nó_prompt_conversacional = PromptNode(
        "gpt-3.5-turbo",
        api_key=openai_api_key,
        max_length=256,
        stop_words=["Observação:"],
    )
    memoria = ConversationSummaryMemory(
        nó_prompt_conversacional,
        prompt_template="deepset/conversational-summary",
        summary_frequency=3,
    )
    prompt_agente = """
    Aqui vai o script completo. Sinta-se à vontade para fazer ajustes conforme necessário.
    """
    return ConversationalAgent(
        prompt_node=nó_prompt_conversacional,
        memory=memoria,
        prompt_template=prompt_agente,
        tools=[ferramenta_pesquisa],
    )

# Inicializar o recuperador e o agente
recuperador = obter_recuperador()
agente = obter_agente(recuperador)
cl.HaystackAgentCallbackHandler(agente)

# Função para renomear o autor
@cl.author_rename
def renomear(orig_author: str):
    dict_rename = {"custom-at-query-time": "Passo do Agente"}
    return dict_rename.get(orig_author, orig_author)

# Função para iniciar a conversa
@cl.on_chat_start
async def iniciar():
    pergunta = "Como era a estátua de Rodes?"
    await cl.Message(author="Usuário", content=pergunta).send()
    resposta = await cl.make_async(agente.run)(pergunta)
    await cl.Message(author="Agente", content=resposta["answers"][0].answer).send()

# Função para responder às mensagens do usuário
@cl.on_message
async def responder(mensagem: cl.Message):
    resposta = await cl.make_async(agente.run)(mensagem.content)
    await cl.Message(author="Agente", content=resposta["answers"][0].answer).send()
