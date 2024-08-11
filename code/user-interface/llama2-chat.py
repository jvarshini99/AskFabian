#pip install langchain llama-cpp-python chainlit

# https://huggingface.co/TheBloke/Airoboros-L2-7B-2.1-GGUF/tree/main

import chainlit as cl

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
import tracemalloc

tracemalloc.start()

MODEL_PATH = "/Users/praneshjayasundar/Documents/Gunner/Boston-University/Fall-2023/student/CS505/final-project/health-assistant/huggingface-models/llama2-trained-medical-v2.Q4_K_M.gguf"

llm = LlamaCpp(
        model_path=MODEL_PATH,
        # n_batch=n_batch,
        # n_ctx=1024,
        temperature=0.3,
        max_tokens=10000,
        n_threads=16,
        verbose=False,  # Verbose is required to pass to the callback manager
        streaming=True,
    )

template = """
    You are AskFabian, a virtual assistant specialized in health and wellness. Your main function is to provide answers to health-related inquiries, including medication guidance, symptom analysis, and emotional support. It is crucial to accurately interpret the user's emotional tone and adjust your responses to match. Your approach should be conversational, empathetic, and precise, ensuring each user's query is addressed with care and understanding.
    In addition to offering accurate health advice, you're also equipped to lighten the mood with humor when appropriate. If a user appears down or in need of a lift, tactfully include light-hearted, appropriate jokes or comical comments to brighten their day, while still providing helpful and relevant information.

    Question: {question}

    Only return the helpful answer below and nothing else. Please use separate lines for more clarity.
    """

# How to react when connection is established
@cl.on_chat_start
async def main():

    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to AskFabian. How can I assist you today?"
    await msg.update()

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")

    # res = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    # print("Result : ",res)

    # await cl.Message(content=res["text"]).send()
    cb = cl.LangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )

    cb.answer_reached = True

    res = await cl.make_async(llm_chain)(message.content, callbacks=[cb])