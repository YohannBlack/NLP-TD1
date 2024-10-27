import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer

def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df['is_comic']

    vectorize = CountVectorizer()
    X = vectorize.fit_transform(df['video_name'])

    X = pd.DataFrame(X.toarray(), columns=vectorize.get_feature_names_out())
    return X, y