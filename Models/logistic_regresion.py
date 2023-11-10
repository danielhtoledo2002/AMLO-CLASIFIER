
# Tool from text processing  
import nltk
from nltk.corpus import stopwords


# Tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
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
  
def clasification_to_num(text):
    if text == 'exterior':
        return 0
    elif text == 'economia':
        return 1
    elif text == 'opinion' :
        return 2
    elif text == 'competencia':
        return 3
    elif text == 'apoyo':
        return 4
    elif text == 'seguridad':
        return 5
  
df['cla_num'] = df['Clasificacion'].apply(clasification_to_num)


def logistic_regresion():
    df['Texto_limpio'] = df['Texto'].apply(clean_text)
    X = df['Texto_limpio']
    y = df['cla_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)
    tfidf = TfidfVectorizer(ngram_range=(1, 3))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    return classification_report(y_test, y_pred)

print(logistic_regresion())