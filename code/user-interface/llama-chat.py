from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationBufferMemory
import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from chainlit.input_widget import TextInput
import tracemalloc

tracemalloc.start()

MODEL_PATH = "/Users/praneshjayasundar/Documents/Gunner/Boston-University/Fall-2023/student/CS505/final-project/health-assistant/huggingface-models/llama2-trained-medical-v2.Q4_K_M.gguf"

@cl.cache
def instantiate_llm():
    n_batch = (
        2048  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    )

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_batch=n_batch,
        n_ctx=2048,
        temperature=0.3,
        max_tokens=10000,
        n_threads=16,
        verbose=True,  # Verbose is required to pass to the callback manager
        streaming=True,
    )
    return llm


llm = instantiate_llm()

add_llm_provider(
    LangchainGenericProvider(id=llm._llm_type, name="Llama-cpp", llm=llm, is_chat=False)
)

@cl.on_chat_start
async def main():
    template = """
    You are AskFabian, a virtual assistant specialized in health and wellness. Your main function is to provide answers to health-related inquiries, including medication guidance, symptom analysis, and emotional support. It is crucial to accurately interpret the user's emotional tone and adjust your responses to match. Your approach should be conversational, empathetic, and precise, ensuring each user's query is addressed with care and understanding.
    In addition to offering accurate health advice, you're also equipped to lighten the mood with humor when appropriate. If a user appears down or in need of a lift, tactfully include light-hearted, appropriate jokes or comical comments to brighten their day, while still providing helpful and relevant information.

    Context: {history}
    Question: {input}

    Only return the helpful answer below and nothing else.
    """

    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to AskFabian. How can I assist you today?"
    await msg.update()

    prompt = PromptTemplate(template=template, input_variables=["history", "input"])

    conversation = LLMChain(
        prompt=prompt, llm=llm, memory=ConversationBufferWindowMemory(k=5)
    )

    cl.user_session.set("conv_chain_v1", conversation)

@cl.on_message
async def main(message: cl.Message):
    conversation = cl.user_session.get("conv_chain_v1")

    cb = cl.LangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )

    cb.answer_reached = True

    res = await cl.make_async(conversation)(message.content, callbacks=[cb])