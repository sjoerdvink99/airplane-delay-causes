import pandas as pd
import streamlit as st
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score, calinski_harabasz_score, precision_score, recall_score, f1_score


# Cached loading function for dataframes
@st.cache
def import_data(path):
    data = pd.read_csv(path)
    return data

# Cached function for applying PCA
@st.cache
def apply_pca(dataframe):
    X = scale(dataframe) # Scaling before PCA
    
    # Source: Hands-on machine learning with scikit-learn, keras & tensorflow by Aurélien Géron (book)
    n_baches = 30 # Defining number of batches
    inc_pca = IncrementalPCA(n_components=3) # Using incremental PCA because of speed
    for X_batch in np.array_split(X, n_baches):
        inc_pca.partial_fit(X_batch)
    X_reduced = inc_pca.transform(X)
    X_reduced = pd.DataFrame(X_reduced)
    return X_reduced

def calc_cluster_evaluation(X, labels):
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    return [round(silhouette, 3), round(calinski, 1)]

def calc_classification_evaluation(y_test, predictions, **kwargs):
    precision = precision_score(y_test, predictions, **kwargs)
    recall = recall_score(y_test, predictions, **kwargs)
    f1 = f1_score(y_test, predictions, **kwargs)
    return [round(precision, 3), round(recall, 3), round(f1, 3)]
