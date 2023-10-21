import pandas as pd
import nltk
from nltk.corpus import stopwords

df = pd.read_csv("amlo.csv")
spanish_stopwords = stopwords.words("spanish")
