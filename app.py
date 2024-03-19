from llm_api import openai_chatbot_chain
import chainlit as cl

#|--------------------------------------------------------------------------|
#|                            On Boarding                                   |
#|--------------------------------------------------------------------------|
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    app_user = cl.user_session.get("user")
    await cl.Message(f"Hello User").send()

#|--------------------------------------------------------------------------|
#|                               Chat                                       |
#|--------------------------------------------------------------------------|
@cl.on_message
async def main(user_input: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": user_input.content})

    llm_output = cl.Message(content="")
    await llm_output.send()

    stream = await openai_chatbot_chain(message_history)

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await llm_output.stream_token(token)

    message_history.append({"role": "assistant", "content": llm_output.content})
    await llm_output.update()