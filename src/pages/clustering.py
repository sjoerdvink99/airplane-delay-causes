import utils
import visualization
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import scale

def app():
    # Loading in the data & apply PCA
    df = utils.import_data('../../data/encoded_airline_delay_dataframe.csv')
    data = utils.apply_pca(df)

    # Model selection
    cluster_model = st.sidebar.selectbox('Select the model', ['Introduction', 'KMeans', 'DBSCAN', 'Linkage based clustering'])

    # Generate model output
    if cluster_model == 'Introduction':
        st.markdown("""
            # Clustering algorithms
            Cluster algorithms are used to discover possible clusters in the dataset. The purpose here is to distinguish clusters such that objects within the same cluster shall be as similar as possible and objects of different clusters shall be as dissimilar as possible. The web application contains 3 cluster algorithms: 
            - KMeans
            - DBSCAN
            - Linkage based

            ## Dimensionality reduction
            Prior to each cluster analysis, a principal component analysis (PCA) is applied to reduce the number of dimensions. An incremental PCA is used to increase the speed of the analysis.
        """)
    
    elif cluster_model == 'KMeans':
        st.markdown("""
            # KMeans
            This algorithm arbitrarily puts cluster representatives in the data, after which they move to the mean of the data they are closest to. After applying PCA, the following clusters start to emerge. 
            ## KMeans visualization
        """)

        # Define the clusters
        n_clusters = st.sidebar.slider('Number of clusters', 2, 10)
        KMeans_model = KMeans(n_clusters=n_clusters).fit(data)
        KMeans_labels = KMeans_model.predict(data)

        # Calculate the evaluation metrics
        silhouette, calinski = utils.calc_cluster_evaluation(data, KMeans_labels)
        col1, col2 = st.columns(2)
        col1.metric("Silhouette", silhouette)
        col2.metric("Calinski", calinski)

        # Plot the data in a scatter plot seperated by color based on labels
        fig = visualization.kmeans_plot(data, KMeans_labels)
        st.write(fig)

    elif cluster_model == 'DBSCAN':
        st.markdown("""
            # Density-Based Clustering of Applications with Noise (DBSCAN)
            DBSCAN groups together points that are close to each other based on the epsilon (eps) and the minimal samples in the eps. After applying PCA, the following clusters start to emerge. 
            ## DBSCAN visualizatie
        """)

        # Retrieve the desired metrics
        st.sidebar.subheader('Chart settings')
        eps = st.sidebar.slider('Eps', 0.1, 2.0)
        min_samples = st.sidebar.slider('Minimal samples', 1, 20)

        # Define the clusters
        DBSCAN_model = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        DBSCAN_labels = DBSCAN_model.labels_

        # Calculate the evaluation metrics
        silhouette, calinski = utils.calc_cluster_evaluation(data, DBSCAN_labels)
        col1, col2 = st.columns(2)
        col1.metric("Silhouette", silhouette)
        col2.metric("Calinski", calinski)

        # Plot the model in a 3D scatter plot
        fig = px.scatter(data, x=0, y=1, color=DBSCAN_labels, title='DBSCAN clustering')
        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
        st.write(fig)

    elif cluster_model == 'Linkage based clustering':
        st.markdown("""
            # Linkage based clustering
            This algorithm merges clusters with minimum distance to a dendogram. After applying PCA, the following clusters start to emerge.
            ## LBC visualizatie
        """)

        # Retrieve the desired metrics
        st.sidebar.subheader('Chart settings')
        n_clusters = st.sidebar.slider('Number of clusters', 2, 10)
        linkage = st.sidebar.selectbox('Linkage type', ['ward', 'complete', 'average', 'single'])

        # Define the clusters
        LBC_model = AgglomerativeClustering(distance_threshold=None, compute_distances=True, linkage=linkage, n_clusters=n_clusters)
        LBC_model.fit(data)
        LBC_labels = LBC_model.labels_
        
        # Calculate the evaluation metrics
        silhouette, calinski = utils.calc_cluster_evaluation(data, LBC_labels)
        col1, col2 = st.columns(2)
        col1.metric("Silhouette", silhouette)
        col2.metric("Calinski", calinski)

        # Plot the model in a plot
        fig = px.scatter(data, x=0, y=1, color=LBC_labels, title='Linkage based clustering')
        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
        st.write(fig)

        fig = visualization.plot_dendrogram(LBC_model, truncate_mode='level', p=4)
        st.pyplot()
