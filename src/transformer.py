import time
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score

def transformer(texts_test,y_test):
    t0 = time.time()

    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt",
        device=-1,
    )

    results = clf(
        list(texts_test),
        truncation=True,
        max_length=128,
        batch_size=32,
    )

    y_pred_transformer = [1 if r["label"] == "NEGATIVE" else 0 for r in results]

    t_transformer = time.time() - t0

    # 5. Metrics
    acc_transformer = accuracy_score(y_test, y_pred_transformer)
    f1_transformer = f1_score(y_test, y_pred_transformer, average="macro")

    return acc_transformer,f1_transformer,t_transformer,y_pred_transformer