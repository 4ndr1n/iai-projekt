import pandas as pd

def eval(acc_classical,f1_classical,t_classical,acc_transformer,f1_transformer,t_transformer,RESULTS_DIR):
    results = pd.DataFrame([
    {
        "model": "classical",
        "accuracy": acc_classical,
        "macro_f1": f1_classical,
        "time_s": t_classical,
    },
    {
        "model": "transformer",
        "accuracy": acc_transformer,
        "macro_f1": f1_transformer,
        "time_s": t_transformer,
    },
])

    results.to_csv(RESULTS_DIR / "results.csv", index=False)
    results
