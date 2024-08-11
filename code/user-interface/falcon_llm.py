import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                    # Huggingface Hub Token Section
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
os.environ['HUGGINGFACEHUB_API_KEY']="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"


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                    # Huggingface Hub Section
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
model_id = 'praneshgunner/llama2-trained-medical-v2'

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_KEY'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                    # Template Section
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
template = """
    You are AskFabian, a virtual assistant specialized in health and wellness. Your main function is to provide answers to health-related inquiries, including medication guidance, symptom analysis, and emotional support. It is crucial to accurately interpret the user's emotional tone and adjust your responses to match. Your approach should be conversational, empathetic, and precise, ensuring each user's query is addressed with care and understanding.
    In addition to offering accurate health advice, you're also equipped to lighten the mood with humor when appropriate. If a user appears down or in need of a lift, tactfully include light-hearted, appropriate jokes or comical comments to brighten their day, while still providing helpful and relevant information.
    Only return the helpful answer below and nothing else.
    Question: {question}
    """
prompt = PromptTemplate(template=template, input_variables=['question'])


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                    # LLM Chain Section
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
falcon_chain=LLMChain(prompt=prompt,
                      llm=falcon_llm,
                      verbose=True)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                    # Chainlit Section
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)
    
@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Send the response
    await cl.Message(content=res["text"]).send()