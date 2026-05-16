import urllib.request
import pandas as pd

from pathlib import Path

def load(DATA_DIR):
    SMS_PATH = DATA_DIR / "sms.tsv"
    MIRRORS = [
        "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
        "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv",
    ]
    state = download_sms(SMS_PATH, MIRRORS)
    
    if state == False:
        print('Data could not be downloaded from Mirrors, exiting')
        exit
    else:
        print('Data is present, analyzing...')
    
    df = load_sms(SMS_PATH)
    return df

def download_sms(target: Path, MIRRORS) -> bool:
    # 1. If `target` already exists and is large enough (>50 KB), return True early
    if target.exists() and target.stat().st_size > 50_000:
        return True

    # 2. Loop over MIRRORS, try each one with a User-Agent header.
    for link in MIRRORS:
        try:
            req = urllib.request.Request(link, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response:
                target.write_bytes(response.read())
            return True
        except Exception as e:
            print(e)
            continue
    if not target.exists():
        return False
    # 3. Return True on success, False if all mirrors fail.


def load_sms(path: Path) -> pd.DataFrame:
    # TODO:
    # 1. Use pd.read_csv with sep="\t", header=None, names=["label", "text"],
    #    encoding="latin-1" (the dataset has some non-UTF-8 characters).
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="latin-1")

    # 2. Add a column `y` that is 1 for spam, 0 for ham.
    df["y"] = df["label"].map({"ham": 0, "spam": 1})
    
    # 3. Return df[["text", "y"]] with NaNs dropped.
    return df[["text", "y"]].dropna()

