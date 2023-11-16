# Manage enviroment  variables
import os

# Tool
import pandas as pd
# web framework
import streamlit as st
# OPEN AI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from Models import logistic_regresion, svccc, tsne, treee, random_forrr

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
    
    st.table(logistic_regresion.clasification_rep())
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
    st.table(svccc.clasification_rep())

    text2 = st.text_input(
        "",
        label_visibility="visible",
        placeholder="Input texto to clasify ",
        key="input2",
    )
    if st.button("Enviar", key="button2"):
        if text2 != "":
            proba = svccc.predict_text(text2)
            st.write(svccc.predict(proba))
st.write("### Desicition Trees")
with st.spinner("Loading table"):
    st.table(treee.clasification_rep())
    text3 = st.text_input(
        "",
        label_visibility="visible",
        placeholder="Input texto to clasify ",
        key="input3",
    )
    if st.button("Enviar", key="button3"):
        if text3 != "":
            proba = treee.predict_text(text3)
            st.write(treee.predict(proba))
st.write("### Random Forest")
with st.spinner("Loading table"):
    st.table(random_forrr.clasification_rep())
    text4 = st.text_input(
        "",
        label_visibility="visible",
        placeholder="Input texto to clasify ",
        key="input4",
    )
    if st.button("Enviar", key="button4"):
        if text4 != "":
            proba = random_forrr.predict_text(text4)
            st.write(random_forrr.predict(proba))
