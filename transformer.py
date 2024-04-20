from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

X_train = pd.read_csv('Train_data.csv')

trf = ColumnTransformer(transformers = [('ohe',OneHotEncoder(sparse_output=False,drop = 'first',dtype=np.int32),['batting_team','bowling_team','city'])
                        ],remainder='passthrough')

trf.fit(X_train)

def transform(X_train):
    X_trf = trf.transform(X_train)

    return X_trf

# X_trf = transform(X_train)
#
# print(X_trf.shape)

