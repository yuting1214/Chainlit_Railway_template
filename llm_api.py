import streamlit as st
from dotenv import load_dotenv
from minha_script import clear_convo, init
from typing import List, Dict
from openai import AsyncOpenAI

# Carregar variáveis de ambiente
load_dotenv('.env')

# Configurações do modelo
model_config = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

# Função para integração com a OpenAI
async def openai_chatbot_chain(messages: List[Dict[str, str]], settings: dict = model_config):
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    stream_response = await client.chat.completions.create(
        messages=messages, stream=True, **settings
    )
    return stream_response

# Inicialização da interface do usuário
init()

# Componentes do Streamlit
st.title("Agente RAG :robot_face:")
api_key = st.sidebar.text_input("Chave da API OpenAI", type="password")
st.sidebar.markdown("""Este aplicativo demonstra o Agente RAG. É capaz de rotear a consulta do usuário para a escolha apropriada 
de resumir um documento, fornecer informações adicionais de um banco de dados vetorial ou fornecer uma simples resposta de acompanhamento.
O agente em si não depende de nenhum orquestrador (por exemplo: llama-index, langchain, etc.) e usa apenas haystack-ai para indexar e recuperar documentos.""")

# Atualizar a chave da API OpenAI
openai.api_key = api_key

# Botão para limpar a conversa
clear_button = st.sidebar.button("Limpar Conversa", key="clear", on_click=clear_convo)

# Entrada do usuário
user_input = st.text_input("Diga algo")

# Lógica de conversa
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    res = await openai_chatbot_chain([{"role": "user", "content": user_input}])
    st.session_state.messages.append({"role": "assistant", "content": res["choices"][0]["message"]["content"]})

# Exibir mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
