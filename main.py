# Manage enviroment  variables
import os

# Tool
import pandas as pd
# web framework
import streamlit as st
# OPEN AI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from Models import logistic_regresion, random_forrr, svccc, treee, tsne






# st.set_page_config(layout='wide')
# start static  web page
df = pd.read_csv("amlo.csv")

st.set_option('deprecation.showPyplotGlobalUse', False)
with st.sidebar:
    st.write(" # Configuration")
    selected = st.multiselect(
        "Columns Logistic Regresion",
        logistic_regresion.clasification_rep().columns,
        default=["Clasificación", "Precision"],
    )
    first_table = logistic_regresion.clasification_rep()[selected]
    selected = st.multiselect(
        "Columns SVG",
        svccc.clasification_rep().columns,
        default=["Clasificación", "Precision"],
    )
    second_table = svccc.clasification_rep()[selected]

    selected = st.multiselect(
        "Columns Decision Tree",
        treee.clasification_rep().columns,
        default=["Clasificación", "Precision"],
    )
    thrird_table = treee.clasification_rep()[selected]
    selected = st.multiselect(
        "Columns Random Forest",
        random_forrr.clasification_rep().columns,
        default=["Clasificación", "Precision"],
    )
    fourt_table = random_forrr.clasification_rep()[selected]


st.write("# AMLO CLASIFIER")

st.write("### Number of clasification")
with st.spinner("Loadig"):
    st.bar_chart(df['Clasificacion'].value_counts(), color="#4A4646")

    
st.write("### TSNE")
# left_co, cent_co,last_co = st.columns(3)
# with cent_co:
st.write("### Word cloud")
with st.spinner("Loading chart"):
    st.plotly_chart(tsne.plot_tsne())

st.write("### Word Cloud")

with st.spinner("Loading"):
    st.image("word_cloud.png", use_column_width=True)



st.write("### Logisic Regresion")
with st.spinner("Loading table"):
    st.dataframe(first_table, hide_index=True, use_container_width=True)
    text = st.text_input(
        "Input text to clasify with Logistic Regresion",
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
    st.dataframe(second_table, hide_index=True, use_container_width=True)
    text2 = st.text_input(
        "Input text to clasify with SVG",
        label_visibility="visible",
        placeholder="Input texto to clasify ",
        key="input2",
    )
    if st.button("Enviar", key="button2"):
        if text2 != "":
            proba = svccc.predict_text(text2)
            st.write(svccc.predict(proba))
st.write("### Desicion Trees")
with st.spinner("Loading table"):
    st.dataframe(thrird_table, hide_index=True, use_container_width=True)
    text3 = st.text_input(
        "Input text with Desicion Tree",
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
    st.dataframe(fourt_table, hide_index=True, use_container_width=True)
    text4 = st.text_input(
        "Input text to clasify with Random Forest",
        label_visibility="visible",
        placeholder="Input texto to clasify ",
        key="input4",
    )
    if st.button("Enviar", key="button4"):
        if text4 != "":
            proba = random_forrr.predict_text(text4)
            st.write(random_forrr.predict(proba))
