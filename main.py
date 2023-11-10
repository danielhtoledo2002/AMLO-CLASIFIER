# Manage enviroment  variables
import os

# Tool from text processing  
import nltk
from nltk.corpus import stopwords

# web framework 
import streamlit as st

# OPEN AI 
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

# Tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# Tool  
import spacy
import pandas as pd
import re
from Models import tsne, logistic_regresion


# start static  web page 

with st.sidebar:
    api_key_file = st.file_uploader('Sube aquí tu Key',
                                    type=['txt'])
    

st.write('# AMLO CLASIFIER')

st.write('### TSNE')

with st.spinner('Loading chart'):
    st.plotly_chart(tsne.plot_tsne() )

st.write("Logisic table")
with st.spinner('Loading chart'):
    st.write(logistic_regresion.logistic_regresion())


if api_key_file is not None:
    key = str(api_key_file.readline().decode('utf-8'))
    os.environ['OPENAI_API_KEY'] = key

    llm = ChatOpenAI(model='gpt-3.5-turbo')
    query = st.text_input("Enter your input text")
    prompt = ''' 
    You are a virtual assistant that can only classify a text in spanish. The classifications are: security, history and 
    economy, but only give me one classification word in spanish
    '''
    if st.button('Generate Output'):
        response = llm.invoke([SystemMessage(content=prompt) ,HumanMessage(content=query)])
        st.write(f"pregunta = {query}")
        st.write(f"## {response.content}")

