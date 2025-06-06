
import os
from dotenv import load_dotenv
from typing import List, Dict
from openai import AsyncOpenAI

# Environment variables
_ = load_dotenv('.env')

model_config = {
    "model": "gpt-4o-mini"
}

async def openai_chatbot_chain(messages: List[Dict[str, str]], settings: dict = model_config):
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    stream_response = await client.chat.completions.create(
        messages=messages, stream=True, **settings
    )
    return stream_response
