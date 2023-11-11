from Models import text_procesing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df  = text_procesing.return_dataframe()

tfidf = TfidfVectorizer(ngram_range=(1, 3))

def logistic_regresion():
    X = df['Texto_limpio']
    y = df['cla_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    model = LogisticRegression(multi_class='multinomial', solver='saga')
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    return model 
    #return classification_report(y_test, y_pred)

model = logistic_regresion()

def predict_text(text):
    resultado = text_procesing.clean_text(text)
    resultado = tfidf.transform([resultado])
    prediccion = model.predict(resultado)
    probabilida = model.predict_proba(resultado)
    return probabilida

def match_category(category):
    match category:
        case 0: return "exterior"
        case 1: return "economia"
        case 2 : return "opinion"
        case 3 : return "competencia"
        case 4: return "apoyo"
        case 5: return "seguridad"

def predict(proba):
    proba = list(proba[0])
    maxx = max(proba)
    index = proba.index(maxx)
    return f"La probabilidad es {maxx} y lo categoriza como {match_category(index)}"


   


