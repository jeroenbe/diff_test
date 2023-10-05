import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import torch



def preprocess_mixed(X, list_categorical_features):
    list_columns = []
    #Preprocess the data so that the categorical variables are from 0 to n_classes - 1
    le = LabelEncoder()
    running_col_idx = 0
    for col_idx in range(X.shape[1]):
        if col_idx in list_categorical_features:
            #if the feature is categorical, encode the column to get values from 0 to n_classes - 1
            X[:, col_idx] = le.fit_transform(X[:, col_idx])
            n_classes = np.unique(X[:, col_idx]).shape[0]
            list_columns.append([running_col_idx + i for i in range(n_classes)])
            running_col_idx += n_classes
        else:
             #if the feature is continuous, then just add it to list_columns.
            list_columns.append([running_col_idx])
            running_col_idx += 1
    #Recall that for each categorical feature i, list_columns[i] will contain the indices of the columns corresponding to the one-hot encoding of the categorical feature
    X_preprocessed = torch.tensor(X)
    
    return X, X_preprocessed, list_columns

def one_hot_mixed(X, list_categorical_features, args):
    X_onehot = np.zeros((args.n,0))
    for col_idx in range(X.shape[1]):
        if col_idx in list_categorical_features:
            encoder = OneHotEncoder()
            transformed_column = encoder.fit_transform(X[:, col_idx].reshape(-1, 1)).toarray()
      
        else:
            transformed_column = X[:, col_idx].reshape(-1,1)
        X_onehot = np.hstack((X_onehot, transformed_column))
    return X_onehot