from langchain.llms import CTransformers
from ctransformers import AutoModelForCausalLM
from time import time
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

import timeit
import argparse
import yaml
import box
from llm.wrapper import setup_qa_chain
from llm.wrapper import query_embeddings

import io
from PIL import Image
import glob
import os
from ingest import run_ingest

custom_prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


#chainlit code
@cl.on_chat_start
async def start():
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to DocEdge GPT. What is your query?"
    await msg.update()

    # Clean the temp directory
    temp_files = glob.glob('temp/*')
    for f in temp_files:
        os.remove(f)

    # Fetching the parameters from the config file 
    with open('dev_scripts/config.yml', 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))

    # Accept input response from Chainlit Bot
    files = None
    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a file to begin!", accept=["application/pdf","image/png","image/jpg","image/jpeg"]
        ).send()
    filename = files[0].name
    with open('data/'+filename, 'wb') as temp_file:
        temp_file.write(files[0].content)

    # Let the user know that the system is ready
    #msg.content = f"Loading input file `{files[0].name}`"
    msg = cl.Message(content=f"Loading input file `{files[0].name}`")
    await msg.send()

    # Ingesting the input file data for creating vectors
    run_ingest()

    # Let the user know that the system is ready
    msg = cl.Message(content=f"The input file {files[0].name} is loaded successfully")
    await msg.send()
    #await msg.update()

    # Let the user know that the system is ready
    #msg.content = 
    #await msg.update()
    msg = cl.Message(content=f"Processing for`{files[0].name}` has started. Please wait for a moment")
    await msg.send()

    # Setting up vector database for input document
    qa_chain = setup_qa_chain()

    # Let the user know that the system is ready
    #msg.content = f"Processing for`{files[0].name}` is done. You can now ask questions!"
    #await msg.send()
    msg = cl.Message(content=f"Processing for`{files[0].name}` is done. You can now ask questions!")
    await msg.send()

    cl.user_session.set("chain", qa_chain)

    # Sending an image with the local file path
    #image1 = cl.Image(name="image1", display="inline", path=r"D:\Pytorch\Pytorch_New_Repo\Pytorch_Yolo_v5\input_images\0001.jpg")
    #await image1.send()

@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    Args:
        message: The user's message.

    Returns:
        None.
    """
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])

    # Send the final answer
    await cl.Message(content=res["result"]).send()