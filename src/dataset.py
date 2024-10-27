import pandas as pd

def make_dataset(filename) -> pd.DataFrame:
    return pd.read_csv(filename)