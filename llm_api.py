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

# Load environment variables
_ = load_dotenv('.env')

# OpenAI model configuration
model_config = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 256,
    "stop_words": ["Observation:"],
}

# Function to interact with the OpenAI chatbot asynchronously
async def openai_chatbot_chain(messages: List[Dict[str, str]], settings: dict = model_config):
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    stream_response = await client.chat.completions.create(
        messages=messages, stream=True, **settings
    )
    return stream_response

# Function to get the document retriever
@cl.cache
def get_retriever():
    document_store = InMemoryDocumentStore(use_bm25=True)
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    document_store.write_documents(dataset)
    return BM25Retriever(document_store)

# Function to get the conversational agent
@cl.cache
def get_agent(retriever):
    pipeline = DocumentSearchPipeline(retriever)
    search_tool = Tool(
        name="seven_wonders_search",
        pipeline_or_node=pipeline,
        description="useful for when you need to answer questions about the seven wonders of the world: Colossus of Rhodes, Statue of Zeus, Great Pyramid of Giza, Mausoleum at Halicarnassus, Temple of Artemis, Lighthouse of Alexandria, and Hanging Gardens of Babylon",
        output_variable="documents",
    )
    conversational_agent_prompt_node = PromptNode(
        "gpt-3.5-turbo",
        api_key=os.getenv('OPENAI_API_KEY'),
        max_length=256,
        stop_words=["Observation:"],
    )
    memory = ConversationSummaryMemory(
        conversational_agent_prompt_node,
        prompt_template="deepset/conversational-summary",
        summary_frequency=3,
    )
    agent_prompt = """
    Here goes the complete script. Feel free to make adjustments as needed.
    """
    return ConversationalAgent(
        prompt_node=conversational_agent_prompt_node,
        memory=memory,
        prompt_template=agent_prompt,
        tools=[search_tool],
    )

# Initialize the retriever and the agent
retriever = get_retriever()
agent = get_agent(retriever)
cl.HaystackAgentCallbackHandler(agent)

# Function to rename the author
@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"custom-at-query-time": "Agent Step"}
    return rename_dict.get(orig_author, orig_author)

# Function to start the conversation
@cl.on_chat_start
async def init():
    question = "What did Rhodes Statue look like?"
    await cl.Message(author="User", content=question).send()
    response = await cl.make_async(agent.run)(question)
    await cl.Message(author="Agent", content=response["answers"][0].answer).send()

# Function to respond to user messages
@cl.on_message
async def answer(message: cl.Message):
    response = await cl.make_async(agent.run)(message.content)
    await cl.Message(author="Agent", content=response["answers"][0].answer).send()
