import numpy as np
import pandas as pd
import spacy 
from spacy import displacy 

nlp = spacy.load("es_core_news_lg")
import nltk
import re
from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words_en= stopwords.words('spanish')
from nltk.stem.wordnet import WordNetLemmatizer

df = pd.read_csv("../OpenAi/amlo_clasify_chatpgt3.csv")
if type(df['vector'][0]) == "str":
    df['vector'] = df["vector"].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))


lemmatizer = WordNetLemmatizer()


def clean(texto):
    texto = texto.lower()
    texto = re.sub(r'([^0-9A-Za-z-À-ÿ \t])','', texto)
    texto = word_tokenize(texto)
    texto = [palabra for palabra in texto if palabra not in stop_words_en]
    texto = [lemmatizer.lemmatize(palabra) for palabra in texto]
    texto = ' '.join(texto)
    return texto

def vectorize_clean(texto):
    texto = texto.lower()
    texto = re.sub(r'([^0-9A-Za-z-À-ÿ \t])','', texto)
    texto = word_tokenize(texto)
    texto = [palabra for palabra in texto if palabra not in stop_words_en]
    texto = [lemmatizer.lemmatize(palabra) for palabra in texto]
    texto = ' '.join(texto)
    texto = nlp(texto).vector
    return texto

def vectorize(texto):
    texto = nlp(texto).vector
    return texto


def getTopXDocs_large(frase,x,export=False):
    data = {
        'texto' :[],
        'sims': []
    }
    buscar = vectorize_clean(frase)
    for vector, frase in zip(df["vector"],df["Texto"]):
        A = buscar 
        B = vector
        resultado = np.dot(A,B) / (np.linalg.norm(A) * np.linalg.norm(B)) 
        
        data["texto"].append(frase)
        data["sims"].append(resultado)
    final = pd.DataFrame(data).sort_values(by = 'sims', ascending = False).head(x)
    return final