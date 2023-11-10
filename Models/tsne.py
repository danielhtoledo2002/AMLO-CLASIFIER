
# Tool from text processing  
import nltk
from nltk.corpus import stopwords


# Tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Tool  
import spacy
import pandas as pd

# Regual expressions
import re


df = pd.read_csv("amlo.csv")


# Stop words in spanish 

stop_words_es = stopwords.words('spanish')

# Load the vacabulary 

nlp = spacy.load('es_core_news_lg')

# fn to clean text 

def clean_text(texto):
  textofin = texto.lower()
  textofin = re.sub(r'([^0-9A-Za-z-À-ÿ \t])','', textofin,)
  textofin = nlp(textofin)
  lema = []
  for token in textofin:
    lema.append(token.lemma_)

  textofin = lema
  textofin = ' '.join(textofin)
  return textofin

# apply the fn to the dataframe 


def plot_tsne():

  df['Texto_limpio'] = df['Texto'].apply(clean_text)

  tfidf = TfidfVectorizer()
  vectorized_text = tfidf.fit_transform(df['Texto_limpio'])

  modelo = TSNE(n_components =2,  init = 'random')
  resultado = modelo.fit_transform(vectorized_text)
  tsne_result_df = pd.DataFrame({'tsne_1': resultado[:,0], 'tsne_2': resultado[:,1]})
  tsne_result_df['label'] = df['Clasificacion']
  tsne_result_df['text'] = df['Texto']
  import plotly.express as px
  fig = px.scatter(data_frame = tsne_result_df,
                  x = tsne_result_df['tsne_1'],
                  y = tsne_result_df['tsne_2'],
                  color = tsne_result_df['label'],
                  template = 'plotly_dark',
                  hover_data = ['text'])
  return fig

