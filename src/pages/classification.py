import utils
import visualization
import streamlit as st
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def app():
    # Generating training and testing split
    df = utils.import_data('../../data/classification_airline_delay_dataframe.csv')
    X, y = df.drop(['ArrDelay'], 1), df['ArrDelay']
    X = scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Model selection
    classification_model = st.sidebar.selectbox('Select the model', ['Introduction', 'KNN', 'Decision tree', 'SVM'])

    # Generate model output
    if classification_model == 'Introduction':
        st.markdown("""
            # Classification algorithms
            Classification means that a certain value can be predicted with the supplied dataset. This can be either a categorical or a numerical value, depending on the algorithm used. The following algorithms are used in this dashboard:
            - K Nearest Neighbors (KNN)
            - Decision tree
            - Support vector machine (SVM)

            ## Predictive values
            This dashboard attempts to classify the arrival delay. In the preprocessing, the arrival delay is divided into two categories, a category with less than 5 minutes delay and a category with more.
        """)
    
    elif classification_model == 'KNN':
        st.markdown("""
            # K-Nearest Neighbors (KNN)
            This algorithm detects k-elements with the shortest distance and classifies a datapoint according to the labels of the nearest neighbors.
            ## KNN predictions
        """)

        # Define the classification
        neighbors = st.sidebar.slider('Number of neighbors', 2, 20)
        knn = KNeighborsClassifier(n_neighbors=neighbors).fit(X_train, y_train)
        knn_predictions = knn.predict(X_test)

        # Calculate the evaluation metrics
        precision, recall, f1 = utils.calc_classification_evaluation(y_test, knn_predictions)
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", precision)
        col2.metric("Recall", recall)
        col3.metric("F1-score", f1)

        # Plot the data
        fig = visualization.create_knn_scatter(X_test, y_test, knn_predictions)
        st.write(fig)
    
    elif classification_model == 'Decision tree':
        st.markdown("""
            # Decision tree
            You want to know the best attributes to predict pure leaf nodes. To find this, you note the count of the label for each distinct value in a column. Apply an attribute selection method to find the best splitting point
            ## Decision tree predictions
        """)

        # Retrieve the desired metrics
        criterion = st.sidebar.selectbox('Attribute selection method', ['gini', 'entropy'])
        max_leave_nodes = st.sidebar.slider('Number of maximum leave nodes', 2, 50)
        show_train = st.sidebar.selectbox('Visualise training data', ['Yes', 'No'])
        show_test = st.sidebar.selectbox('Visualise test data', ['Yes', 'No'])

        # Define the classification
        dtree = DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=max_leave_nodes).fit(X_train, y_train)
        dtree_predictions = dtree.predict(X_test)

        # Calculate the evaluation metrics
        precision, recall, f1 = utils.calc_classification_evaluation(y_test, dtree_predictions)
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", precision)
        col2.metric("Recall", recall)
        col3.metric("F1-score", f1)

        # Plot the data
        dimensions = list(df.columns)
        dimensions.remove('ArrDelay')
        fig = visualization.create_dtree(dtree, dimensions)
        st.pyplot()
        
        fig = visualization.visualise_dtree(X_train, y_train, X_test, y_test, lambda x: dtree.apply(x), show_train, show_test)
        st.pyplot(fig)

    elif classification_model == 'SVM':
        st.markdown("""
            # Support vector machine (SVM)
            The algorithm creates a line or a hyperplane which separates the data into classes, the goal is to find a hyperplane with the biggest margin between the line and the support vectors.
            ## SVM predictions
        """)

        # Retrieve the desired metrics
        kernel = st.sidebar.selectbox('Kernel', ['rbf'])
        show_supports = st.sidebar.selectbox('Visualise support vectors', ['Yes', 'No'])
        show_train = st.sidebar.selectbox('Visualise training data', ['Yes', 'No'])
        show_test = st.sidebar.selectbox('Visualise test data', ['Yes', 'No'])

        # Map the 13D data to 2D
        v = PCA(n_components=2).fit(X_train)
        Xt_train = v.transform(X_train)
        Xt_test = v.transform(X_test)
        Xt = v.transform(X)

        # Define the classification
        svc = SVC(kernel=kernel, gamma='scale').fit(Xt_train, y_train)
        svc_predictions = svc.predict(Xt_test)

        # Calculate the evaluation metrics
        precision, recall, f1 = utils.calc_classification_evaluation(y_test, svc_predictions)
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", precision)
        col2.metric("Recall", recall)
        col3.metric("F1-score", f1)

        x_min = np.min(Xt[:, 0])
        x_max = np.max(Xt[:, 0])
        y_min = np.min(Xt[:, 1])
        y_max = np.max(Xt[:, 1])

        # evaluate decision function in a 200x200 grid
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

        # execute the decision function on the grid
        gridpts = np.array((xx, yy)).T.reshape(-1, 2)
        Z = svc.decision_function(gridpts)
        Z = Z.reshape(xx.shape).T

        # visualize decision function for these parameters
        fig = plt.figure()
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)

        if show_train == "Yes":
            plt.scatter(Xt_train[:, 0], Xt_train[:, 1], c=y_train, cmap=plt.cm.RdBu_r, edgecolors="k")

        if show_test == "Yes":
            plt.scatter(Xt_test[:, 0], Xt_test[:, 1], c=y_test, cmap=plt.cm.RdBu_r, edgecolors="k")

        if show_supports == "Yes":
            support_vectors = svc.support_vectors_
            plt.scatter(support_vectors[:,0], support_vectors[:,1], color='green')

        st.pyplot(fig)
