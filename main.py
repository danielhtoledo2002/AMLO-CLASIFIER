# Manage enviroment  variables
import os
# web framework
import streamlit as st
# OPEN AI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
# Tool
import pandas as pd

from Models import tsne, logistic_regresion
# st.set_page_config(layout='wide')
# start static  web page


with st.sidebar:
    st.write(" # Configuration")

st.write("# AMLO CLASIFIER")
st.write("### TSNE")
# left_co, cent_co,last_co = st.columns(3)
# with cent_co:
with st.spinner("Loading chart"):
    st.plotly_chart(tsne.plot_tsne())

st.write("### Logisic Regresion")
with st.spinner("Loading table"):
    text = st.text_input(
        "", label_visibility="visible", placeholder="Input texto to clasify "
    )
    if st.button("Enviar"):
        if text != "":
            proba = logistic_regresion.predict_text(text)
            st.write(logistic_regresion.predict(proba))






