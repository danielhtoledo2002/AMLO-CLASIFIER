import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

with st.sidebar:
    api_key_file = st.file_uploader('Sube aquí tu Key',
                                    type=['txt'])
st.write('# MI GPT')
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
