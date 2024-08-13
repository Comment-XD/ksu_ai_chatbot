import streamlit as st
from ksu_ai_chatbot import ksu_ai_bot
import time

def stream_response(prompt:str, delay: float | int):
    for word in prompt.split(" "):
        yield word + " "
        time.sleep(delay)
    

main_page = st.container(height=500)
main_page.header("KSU AI Chatbot", anchor="center")
main_page.markdown(
    """***
    Ask questions about the KSU's AI Club, and it shall give you the answers!
    """
)

prompt = st.chat_input("Ask away!")

if prompt:
    time.sleep(1)
    human = main_page.chat_message("human")
    human.write(prompt)
    
    time.sleep(1)
    
    ai = main_page.chat_message("ai")
    rag_response = ksu_ai_bot(prompt)
    ai.write_stream(stream_response(rag_response, 0.05))
    

