import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

try:
    from . import load_data as ld
    from . import classical as cl
    from . import transformer as tr
    from . import evaluate as e
except ImportError:
    import load_data as ld
    import classical as cl
    import transformer as tr
    import evaluate as e

def setup():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    DATA_DIR    = Path("data/raw")
    RESULTS_DIR = Path("results")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR, RESULTS_DIR, SEED


def split_data(df,SEED):
    X = np.asarray(df["text"].tolist())
    y = np.asarray(df["y"].tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    print(f"train: {len(X_train):,}   test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test

def comparison(texts_test,y_test,y_pred_classical,y_pred_transformer):
    disagree_df = pd.DataFrame({
        "text": texts_test,
        "true": y_test,
        "classical": y_pred_classical,
        "transformer": y_pred_transformer,
    })

    disagree_df = disagree_df[disagree_df["classical"] != disagree_df["transformer"]]

    print(f"Disagreements: {len(disagree_df):,} examples")
    print(disagree_df.head(5))

def main():
    DATA_DIR, RESULTS_DIR,SEED = setup()
    df = ld.load(DATA_DIR)

    texts_train, texts_test, y_train, y_test = split_data(df,SEED)
    acc_classical,f1_classical,t_classical,y_pred_classical = cl.regression(texts_train,texts_test,y_train,y_test,SEED)
    acc_transformer,f1_transformer,t_transformer,y_pred_transformer = tr.transformer(texts_test,y_test)

    e.eval(acc_classical,f1_classical,t_classical,acc_transformer,f1_transformer,t_transformer,RESULTS_DIR)
    comparison(texts_test,y_test,y_pred_classical,y_pred_transformer)

if __name__ == '__main__':
    main()