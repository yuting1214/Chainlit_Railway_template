
import os
from dotenv import load_dotenv
from typing import List, Dict
from openai import AsyncOpenAI

# Environment variables
_ = load_dotenv('.env')

model_config = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

async def openai_chatbot_chain(messages: List[Dict[str, str]], settings: dict = model_config):
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    stream_response = await client.create_chat_chain(
        messages=messages, stream=True, **settings
    )
    return stream_response