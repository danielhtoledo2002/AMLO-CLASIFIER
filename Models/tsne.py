

# Tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import pandas as pd
import plotly.express as px

from Models import text_procesing


# apply the fn to the dataframe 

tsne_result_df  =  pd.read_csv("TSNe/tsne_results.csv")
def plot_tsne():
  fig = px.scatter(data_frame = tsne_result_df,
                  x = tsne_result_df['tsne_1'],
                  y = tsne_result_df['tsne_2'],
                  color = tsne_result_df['label'],
                  template = 'plotly_dark',
                  hover_data = ['text'])
  return fig

