import os
# web framework

import pandas as pd
import streamlit as st
# OPEN AI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
# Tool

df = pd.read_csv("amlo.csv")
os.environ['OPENAI_API_KEY'] = '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'
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = """ 
You are a virtual assistant that can only classify a text in spanish. The classifications are: security, history and 
economy, but only give me one classification word in spanish
"""

def clasificate_csv(text):
        
        response = llm.invoke(
                [SystemMessage(content=prompt), HumanMessage(content=text)]
            )
        return response.content 

df["new_clasification"] = df["Texto"].apply(clasificate_csv)
print("acabo")

df.to_csv('amlo_clasify_chatpgt.csv', index=False)


