from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import pandas as pd
import plotly.express as px
import joblib
import text_pro




df  =  text_pro.return_dataframe()
tfidf = TfidfVectorizer()
vectorized_text = tfidf.fit_transform(df['Texto_limpio'])
modelo = TSNE(n_components =2,  init = 'random')
resultado = modelo.fit_transform(vectorized_text)
tsne_result_df = pd.DataFrame({'tsne_1': resultado[:,0], 'tsne_2': resultado[:,1]})
tsne_result_df['label'] = df['classification_spanish']
tsne_result_df['text'] = df['Texto']
joblib.dump(tfidf, "TSNe/tfidf_vectorizer.joblib")
joblib.dump(modelo, "TSNe/tsne_model2.joblib")
tsne_result_df.to_csv("TSNe/tsne_results2.csv", index=False)