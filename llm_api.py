import streamlit as st
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.utils import Secret
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
import concurrent.futures
import os
from pathlib import Path
import openai

from utils.custom_converters import DocxToTextConverter

# Carregar variáveis de ambiente
load_dotenv('.env')

# Inicializar o armazenamento de documentos
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

# Configurações do modelo OpenAI
model_config = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

# Função para escrever documentos no armazenamento
def write_documents(file):
    pipeline = Pipeline()
    if file.name.endswith(".docx"):
        pipeline.add_component("converter", DocxToTextConverter())
    elif file.name.endswith(".txt"):
        pipeline.add_component("converter", TextFileToDocument())
    else:
        pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=350))
    pipeline.add_component("embedder", OpenAIDocumentEmbedder(api_key=Secret.from_token(openai.api_key)))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder.documents", "writer")
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    pipeline.run({"converter": {"sources": [Path(file_path)]}})
    st.success("Documento indexado com sucesso!")

# Função para dividir documentos em partes para sumarização
def chunk_documents(file):
    pipeline = Pipeline()
    if file.name.endswith(".docx"):
        pipeline.add_component("converter", DocxToTextConverter())
    elif file.name.endswith(".txt"):
        pipeline.add_component("converter", TextFileToDocument())
    else:
        pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=3000))
    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    file_path = os.path.join("uploads", file.name)
    docs = pipeline.run({"converter": {"sources": [file_path]}})
    return [d.content for d in docs["splitter"]["documents"]]

# Função para consultar o pipeline de documentos
def query_pipeline(query):
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", OpenAITextEmbedder(Secret.from_token(openai.api_key)))
    query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=4))
    query_pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(document_store=document_store, top_k=4))
    query_pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=4, sort_by_score=True))
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("bm25_retriever", "joiner")
    query_pipeline.connect("retriever", "joiner")
    result = query_pipeline.run({"text_embedder": {"text": query}, "bm25_retriever": {"query": query}})
    return result["joiner"]["documents"]

# Função para rotear a consulta para a escolha apropriada com base na resposta do sistema
def query_router(query):
    generator = OpenAIChatGenerator(api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo")
    system = """Você é um bot de roteamento de consulta profissional para um sistema de chatbot que decide se a consulta de um usuário requer um resumo, uma recuperação de informações extras de um banco de dados vetorial, ou uma resposta simples de saudação/agradecimento/cumprimento. Se a consulta exigir um resumo, você responderá apenas "(1)". Se a consulta exigir informações extras, você responderá apenas "(2)". Se a consulta exigir uma resposta de saudação/agradecimento/cumprimento ou uma resposta a uma pergunta de acompanhamento baseada no histórico da conversa, você responderá apenas "(3)"."""
    instruction = f"""Você recebeu a consulta de um usuário no campo <query>. Você é responsável por rotear a consulta para a escolha apropriada conforme descrito na resposta do sistema. <query>{query}</query> Você também recebeu o histórico da conversa no campo <history>{st.session_state.messages}</history>."""
    messages = [ChatMessage.from_system(system), ChatMessage.from_user(instruction)]
    response = generator.run(messages)
    return response

# Função para sumarizar cada parte do documento com base na consulta do usuário
def map_summarizer(query, chunk):
    generator = OpenAIChatGenerator(api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo")
    system = """Você é um sumarizador de corpora profissional para um sistema de chatbot. Você é responsável por sumarizar uma parte do texto de acordo com a consulta de um usuário."""
    instruction = f"""Você recebeu a consulta de um usuário no campo <query>. Responda adequadamente à entrada do usuário usando a parte fornecida no campo <chunk>: <query>{query}</query>\n <chunk>{chunk}</chunk>"""
    messages = [ChatMessage.from_system(system), ChatMessage.from_user(instruction)]
    response = generator.run(messages)
    return response

# Função para sumarizar a lista de resumos em um resumo final com base na consulta do usuário
def reduce_summarizer(query, analyses):
    generator = OpenAIChatGenerator(api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo")
    system = """Você é um sumarizador de corpora profissional para um sistema de chatbot. Você é responsável por sumarizar uma lista de resumos de acordo com a consulta de um usuário."""
    instruction = f"""Você recebeu a consulta de um usuário no campo <query>. Responda adequadamente à entrada do usuário usando a lista de resumos fornecida no campo <chunk>: <query>{query}</query>\n <chunk>{analyses}</chunk>"""
    messages = [ChatMessage.from_system(system), ChatMessage.from_user(instruction)]
    response = generator.run(messages)
    return response

# Função para responder a uma consulta do usuário com base em uma resposta simples de acompanhamento
def simple_responder(query):
    generator = OpenAIChatGenerator(api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo")
    system = """Você é um profissional de resposta de acompanhamento de saudação/agradecimento/cumprimento para um sistema de chatbot. Você é responsável por responder educadamente a uma consulta de usuário."""
    instruction = f"""Você recebeu a consulta de um usuário no campo <query>. Responda adequadamente à entrada do usuário: <query>{query}</query>"""
    messages = []
    history = st.session_state.messages
    messages.append(ChatMessage.from_system(system))
    for i in range(0, len(history) - 1, 2):
        messages.append(ChatMessage.from_user(history[i]["content"]))
        messages.append(ChatMessage.from_assistant(history[i + 1]["content"]))
    messages.append(ChatMessage.from_user(instruction))
    response = generator.run(messages)
    return response

# Função para sumarizar o documento com base na consulta do usuário
def summary_tool(query, file):
    chunks = chunk_documents(file)
    futures = []
    analyses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for chunk in chunks:
            futures.append(executor.submit(map_summarizer, query, chunk))
        for future in concurrent.futures.as_completed(futures):
            analyses.append(future.result())
        return reduce_summarizer(query, analyses)

# Função para recuperar contexto com base na consulta do usuário
def context_tool(query):
    context = query_pipeline(query)
    context = [c.content for c in context]
    generator = OpenAIChatGenerator(api_key=Secret.from_token(openai.api_key), model="gpt-4-turbo")
    system = """Você é um profissional de resposta de Q/A para um sistema de chatbot. Você é responsável por responder a uma consulta de usuário usando APENAS o contexto fornecido dentro das tags <context>."""
    instruction = f"""Você recebeu a consulta de um usuário no campo <query>. Responda adequadamente à entrada do usuário usando apenas o contexto no campo <context>: <query>{query}</query>\n <context>{context}</context>"""
    messages = [ChatMessage.from_system(system), ChatMessage.from_user(instruction)]
    response = generator.run(messages)
    return response

# Classe do Agente RAG
class RAGAgent:
    def __init__(self):
        self.loops = 0

    def invoke_agent(self, query, file):
        intent = query_router(query)["replies"][0].content.strip()
        if intent == "(1)":
            st.success("Recuperando Resumo...")
            response = summary_tool(query, file)["replies"][0].content
        elif intent == "(2)":
            st.success("Recuperando Contexto...")
            response = context_tool(query)["replies"][0].content
        elif intent == "(3)":
            st.success("Recuperando Resposta Simples...")
            response = simple_responder(query)["replies"][0].content
        return response

# Função para limpar a conversa
def clear_convo():
    st.session_state["messages"] = []

# Função de inicialização
def init():
    st.set_page_config(page_title="GPT RAG", page_icon=":robot_face: ")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

# Inicialização
if __name__ == "__main__":
    init()
    agent = RAGAgent()

    # Componentes do Streamlit
    st.title("Agente RAG :robot_face:")

    api_key = st.sidebar.text_input("Chave da API OpenAI", type="password")
    st.sidebar.markdown(
        """Este aplicativo demonstra o Agente RAG. É capaz de rotear a consulta do usuário para a escolha apropriada 
        de resumir um documento, fornecer informações adicionais de um banco de dados vetorial ou fornecer uma simples resposta de acompanhamento.
        O agente em si não depende de nenhum orquestrador (por exemplo: llama-index, langchain, etc.) e usa apenas haystack-ai para indexar e recuperar documentos."""
    )
    openai.api_key = api_key
    clear_button = st.sidebar.button(
        "Limpar Conversa", key="clear", on_click=clear_convo
    )

    file = st.file_uploader("Escolha um arquivo para indexar...", type=["docx", "pdf", "txt"])
    clicked = st.button("Enviar Arquivo", key="Upload")
    if file and clicked:
        with st.spinner("Aguarde..."):
            write_documents(file)

    user_input = st.text_input("Diga algo")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        res = agent.invoke_agent(user_input, file)
        st.session_state.messages.append({"role": "assistant", "content": res})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
