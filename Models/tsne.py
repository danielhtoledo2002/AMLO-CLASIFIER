

# Tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import pandas as pd
import plotly.express as px

from Models import text_procesing


# apply the fn to the dataframe 

df  =  text_procesing.return_dataframe()


def plot_tsne():


  tfidf = TfidfVectorizer()
  vectorized_text = tfidf.fit_transform(df['Texto_limpio'])

  modelo = TSNE(n_components =2,  init = 'random')
  resultado = modelo.fit_transform(vectorized_text)
  tsne_result_df = pd.DataFrame({'tsne_1': resultado[:,0], 'tsne_2': resultado[:,1]})
  tsne_result_df['label'] = df['Clasificacion']
  tsne_result_df['text'] = df['Texto']

  fig = px.scatter(data_frame = tsne_result_df,
                  x = tsne_result_df['tsne_1'],
                  y = tsne_result_df['tsne_2'],
                  color = tsne_result_df['label'],
                  template = 'plotly_dark',
                  hover_data = ['text'])
  return fig

