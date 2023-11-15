from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

import text_procesing

df = text_procesing.return_dataframe()


X = df["Texto_limpio"]
y = df["cla_num"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.85, random_state=101
)


pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("clf", RandomForestClassifier()),
    ]
)

parameters = {
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 3),
        (1, 3),
    ],
    "clf__n_estimators": [50, 100, 250, 500],
    "clf__criterion": ["entropy", "log_loss"],
    "clf__random_state": [40, 50],
}

grid = GridSearchCV(
    pipeline, parameters, cv=20, n_jobs=-1, verbose=3, scoring="accuracy"
)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
