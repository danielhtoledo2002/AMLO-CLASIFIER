import streamlit as st

import sys 
sys.path.append("Scripts_models")

import random_forest_humano as rf_humano
import random_forest_15k as rf_15k
import random_forest_21k as rf_21k


st.set_option('deprecation.showPyplotGlobalUse', False)

select_clas = ""

with st.sidebar:
    st.write(" # Configuration")
    
    clas = st.radio(
        "Select  which clasification you want to use",
        ["Human", "ChatGPT 15k", "ChatGPT 21k"],
        index=None,
    )
    select_clas = clas

    if select_clas == "Human":
        st.write("in progress")

        selected = st.multiselect(
            "Columns Random Forest",
            rf_humano.clasification_rep().columns,
            default=["Clasificación", "Precision"],
        )
        table1 = rf_humano.clasification_rep()[selected]
    elif select_clas == "ChatGPT 15k":
        st.write("in progress")

        selected = st.multiselect(
            "Columns Random Forest",
            rf_15k.clasification_rep().columns,
            default=["Clasificación", "Precision"],
        )
        table2 = rf_15k.clasification_rep()[selected]
    else: 
        st.write("in progress")

        selected = st.multiselect(
            "Columns Random Forest",
            rf_21k.clasification_rep().columns,
            default=["Clasificación", "Precision"],
        )
        table3 = rf_21k.clasification_rep()[selected]


if select_clas == "Human":
    st.write("# AMLO CLASIFIER (RANDOM FOREST)")
    st.write("# ")

    with st.spinner("Loading table"):
        st.dataframe(table1, hide_index=True, use_container_width=True)
        text1 = st.text_input(
            "Input text with Random Forest",
            label_visibility="visible",
            placeholder="Input text to clasify ",
            key="input1",
        )
        if st.button("Enviar", key="button1"):
            if text1 != "":
                proba = rf_humano.predict_text(text1)
                st.write(rf_humano.predict(proba)) 
elif select_clas == "ChatGPT 15k":
    st.write("# AMLO CLASIFIER (RANDOM FOREST)")
    st.write("# ")
    
    with st.spinner("Loading table"):
        st.dataframe(table2, hide_index=True, use_container_width=True)
        text2 = st.text_input(
            "Input text with Random Forest",
            label_visibility="visible",
            placeholder="Input text to clasify ",
            key="input2",
        )
        if st.button("Enviar", key="button2"):
            if text2 != "":
                proba = rf_15k.predict_text(text2)
                st.write(rf_15k.predict(proba))    
else :
    st.write("# AMLO CLASIFIER (RANDOM FOREST)")
    st.write("# ")

    with st.spinner("Loading table"):
        st.dataframe(table3, hide_index=True, use_container_width=True)
        text4 = st.text_input(
            "Input text with Random Forest",
            label_visibility="visible",
            placeholder="Input text to clasify ",
            key="input3",
        )
        if st.button("Enviar", key="button3"):
            if text4 != "":
                proba = rf_21k.predict_text(text4)
                st.write(rf_21k.predict(proba)) 
