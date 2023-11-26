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
from Models2 import tesne, logistic, svg
from DeepLearningModels import load_CNN, load_FNN





# st.set_page_config(layout='wide')
# start static  web page
df = pd.read_csv("OpenAi/amlo_clasify_chatpgt3.csv")
df2 = pd.read_csv("amlo.csv")

st.set_option('deprecation.showPyplotGlobalUse', False)

select_clas = ""

with st.sidebar:
    st.write(" # Configuration")
    st.write("We train two types of models, one that  was classified by human and other that chat-gpt-3.5 did.")
    clas = st.radio(
        "Select  which clasification you want to use",
        ["Chat gpt :computer:", "Human :male-technologist:"],
        index=None,
    )
    select_clas = clas
    if select_clas == "Chat gpt :computer:":
        st.write("in progress")
        selected = st.multiselect(
            "Columns Logistic Regresion",
            logistic.clasification_rep().columns,
            default=["Clasificación", "Precision"],
        )
        first_table2 = logistic.clasification_rep()[selected]
        
        selected = st.multiselect(
            "Columns SVG",
            svg.clasification_rep().columns,
            default=["Clasificación", "Precision"],
        )
        second_table2 = svg.clasification_rep()[selected]

        selected = st.multiselect(
            "Columns CNN",
            load_CNN.clasification_rep().columns,
            default=["Clasificación", "Precision"],
        )
        third_table2 = load_CNN.clasification_rep()[selected]

        selected = st.multiselect(
            "Columns FNN",
            load_FNN.clasification_rep().columns,
            default=["Clasificación", "Precision"],
        )
        Fourth_table2 = load_FNN.clasification_rep()[selected]
    
    else: 
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



if select_clas == "Chat gpt :computer:":
    st.write("# AMLO CLASIFIER")
    st.write("### Number of clasification")
    with st.spinner("Loadig"):
        st.bar_chart(df['classification_spanish'].value_counts(), color="#4A4646")

    with st.spinner("Loading"):
        st.image("word_cloud2.png", use_column_width=True)


    st.write("### TSNE")
    # left_co, cent_co,last_co = st.columns(3)
    # with cent_co:
    st.write("### ")
    with st.spinner("Loading chart"):
        st.plotly_chart(tesne.plot_tsne())

    st.write("### Logisic Regresion")
    with st.spinner("Loading table"):
        st.dataframe(first_table2, hide_index=True, use_container_width=True)
        text = st.text_input(
            "Input text to clasify with Logistic Regresion",
            label_visibility="visible",
            placeholder="Input texto to clasify ",
            key="input5",
        )
        if st.button("Enviar", key="button7"):
            if text != "":
                proba = logistic.predict_text(text)
                st.write(logistic.predict(proba))

    st.write("### SVC")
    with st.spinner("Loading table"):
        st.dataframe(second_table2, hide_index=True, use_container_width=True)
        text2 = st.text_input(
            "Input text to clasify with SVG",
            label_visibility="visible",
            placeholder="Input texto to clasify ",
            key="input6",
        )
        if st.button("Enviar", key="button6"):
            if text2 != "":
                proba = svg.predict_text(text2)
                st.write(svg.predict(proba))

    st.write("### CNN")
    with st.spinner("Loading table"):
        st.dataframe(third_table2, hide_index=True, use_container_width=True)
        text3 = st.text_input(
            "Input text to clasify with CNN",
            label_visibility="visible",
            placeholder="Input texto to clasify ",
            key="input7",
        )
        if st.button("Enviar", key="button8"):
            if text3 != "":
                proba = load_CNN.predict_text(text3)
                st.write(load_CNN.predict(proba))
    st.write("### FNN")
    with st.spinner("Loading table"):
        st.dataframe(Fourth_table2, hide_index=True, use_container_width=True)
        text4 = st.text_input(
            "Input text to clasify with FNN",
            label_visibility="visible",
            placeholder="Input texto to clasify ",
            key="input8",
        )
        if st.button("Enviar", key="button9"):
            if text4 != "":
                proba = load_FNN.predict_text(text3)
                st.write(load_FNN.predict(proba))
else :
    st.write("# AMLO CLASIFIER")

    st.write("### Number of clasification")
    with st.spinner("Loadig"):
        st.bar_chart(df2['Clasificacion'].value_counts(), color="#4A4646")

        
    st.write("### TSNE")
    # left_co, cent_co,last_co = st.columns(3)
    # with cent_co:
    st.write("### ")
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
