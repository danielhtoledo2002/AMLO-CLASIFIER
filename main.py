# Manage enviroment  variables
import os

# Tool
import pandas as pd
# web framework
import streamlit as st
# OPEN AI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from Models import logistic_regresion, svccc, tsne

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
        "",
        label_visibility="visible",
        placeholder="Input texto to clasify ",
        key="input",
    )
    if st.button("Enviar"):
        if text != "":
            proba = logistic_regresion.predict_text(text)
            st.write(logistic_regresion.predict(proba))

st.write("### SVC")
with st.spinner("Loading table"):
    text2 = st.text_input(
        "",
        label_visibility="visible",
        placeholder="Input texto to clasify ",
        key="input2",
    )
    if st.button("Enviar", key="button2"):
        if text != "":
            proba = svccc.predict_text(text2)
            st.write(svccc.predict(proba))
