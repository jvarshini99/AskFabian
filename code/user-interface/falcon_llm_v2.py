import chainlit as cl
import tracemalloc
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage

tracemalloc.start()

model_path = "/Users/praneshjayasundar/Documents/Gunner/Boston-University/Fall-2023/student/CS505/final-project/health-assistant/huggingface-models/llama2-trained-medical-v2.Q4_K_M.gguf"

@cl.on_chat_start
async def on_chat_start():
    # Initialize your model
    model = LlamaCpp(model_path=model_path)

    # Define the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are AskFabian, a virtual assistant specialized in health and wellness having a conversation with a human. Your main function is to provide answers to health-related inquiries, including medication guidance, symptom analysis, and emotional support. It is crucial to accurately interpret the user's emotional tone and adjust your responses to match. Your approach should be conversational, empathetic, and precise, ensuring each user's query is addressed with care and understanding. In addition to offering accurate health advice, you're also equipped to lighten the mood with humor when appropriate. If a user appears down or in need of a lift, tactfully include light-hearted, appropriate jokes or comical comments to brighten their day, while still providing helpful and relevant information."
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ]
    )

    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Set up the LLMChain
    chat_llm_chain = LLMChain(
        llm=model,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    # Store the LLMChain in the user session
    cl.user_session.set("chat_llm_chain", chat_llm_chain)

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the LLMChain from the user session
    chat_llm_chain = cl.user_session.get("chat_llm_chain")

    # Use the LLMChain to generate a response
    response = chat_llm_chain.predict(human_input=message.content)

    # Send the response
    await cl.Message(content=response).send()
