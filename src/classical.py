from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

def regression(texts_train,texts_test,y_train,y_test,SEED):
    t0 = time.time()
    vec_classical = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_classical = vec_classical.fit_transform(texts_train)
    X_test_classical = vec_classical.transform(texts_test)

    clf_classical = LogisticRegression(max_iter=1000, random_state=SEED)
    clf_classical.fit(X_train_classical, y_train)
    y_pred_classical = clf_classical.predict(X_test_classical)
    t_classical = time.time() - t0

    acc_classical = accuracy_score(y_test, y_pred_classical)
    f1_classical = f1_score(y_test, y_pred_classical, average="macro")

    print(classification_report(y_test, y_pred_classical, target_names=["ham", "spam"]))
    return acc_classical,f1_classical,t_classical, y_pred_classical