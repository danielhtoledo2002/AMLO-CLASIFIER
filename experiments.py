import os
# web framework

import pandas as pd
import streamlit as st
# OPEN AI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
# Tool

df = pd.open_csv("amlo_complete.csv")
os.environ["OPENAI_API_KEY"] = "sk-QMEfMcdEfGjH1LtcqPUdT3BlbkFJJNeyTwQcyKBGKIcMiANli"


llm = ChatOpenAI(model="gpt-3.5-turbo")
query = st.text_input("Enter your input text")
prompt = """ 
You are a virtual assistant that can only classify a text in spanish. The classifications are: security, history and 
economy, but only give me one classification word in spanish
"""
response = llm.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=query)]
    )
st.write(f"pregunta = {query}")
st.write(f"## {response.content}")
