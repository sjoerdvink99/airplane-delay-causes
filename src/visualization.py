import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn import tree
import matplotlib.pyplot as plt

# ----- Introduction visualizations ----- #
@st.cache
def create_flight_map(df, month, delayed):
    df = df.loc[df['Month'] == month]

    if delayed == 0:
        df = df.loc[df['ArrDelay'] == 0]
    else:
        df = df.loc[df['ArrDelay'] > delayed]

    # Source for flight map: https://plotly.com/python/lines-on-maps/
    fig = go.Figure()

    lons = []
    lats = []
    lons = np.empty(3 * len(df))
    lons[::3] = df['origin_long']
    lons[1::3] = df['dest_long']
    lons[2::3] = None
    lats = np.empty(3 * len(df))
    lats[::3] = df['origin_lat']
    lats[1::3] = df['dest_lat']
    lats[2::3] = None

    fig.add_trace(
        go.Scattergeo(
            locationmode = 'USA-states',
            text = df['Month'],
            lon = lons,
            lat = lats,
            mode = 'lines',
            line = dict(width=1, color='red'),
            opacity = 0.5,
        )
    )

    fig.update_layout(
        title_text = 'Domestic flights operated by large air carriers',
        showlegend = False,
        geo = go.layout.Geo(
            scope = 'north america',
            projection_type = 'equirectangular',
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
        ),
    )

    return fig

@st.cache
def create_scatter_ArrDelay(df, type):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    means = []

    for month in range(1, 13):
        mean = df.loc[df.Month == month, type].mean()
        means.append(round(mean))

    fig = px.bar(
        df,
        x=months,
        y=means,
        title='Average delay per month',
        labels={'x': 'Month', 'y':'Average delay (min.)'}
    )
    
    return fig

def unique_airline_hist(df):
    carriers = list(df['UniqueCarrier'].unique())
    undelay_count = []
    delay_count = []
    percentages = []

    for i in carriers:
        undelayed = len(df[(df['UniqueCarrier'] == i) & (df['ArrDelay'] < 10)])
        delayed = len(df[(df['UniqueCarrier'] == i) & (df['ArrDelay'] > 10)])
        undelay_count.append(undelayed)
        delay_count.append(delayed)

    for delayed, undelayed in zip(delay_count, undelay_count):
        percentage = (delayed / (delayed + undelayed)) * 100
        percentages.append(percentage)


    fig = px.bar(
        x=carriers,
        y=percentages,
        title='Percentage of delayed fligth per carrier (more than 10min)',
        labels={'x': 'Carrier', 'y':'Percentage of delayed flights'}
    )

    return fig

# ----- Clustering visualizations ----- #
def kmeans_plot(df, KMeans_labels):
    pc1 = df.iloc[:,0]
    pc2 = df.iloc[:,1]
    fig = px.scatter(x=pc1, y=pc2, color=KMeans_labels, title="KMeans clustering")
    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    return fig

def plot_dendrogram(model, **kwargs):
    # Function from the sklearn documentation
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)

# ----- Classification visualizations ----- #
@st.cache
def create_knn_scatter(X_test, y_test, knn_predictions):
    # Source: https://plotly.com/python/knn-classification/
    fig = px.scatter(
        X_test, x=1, y=2,
        color=knn_predictions,
        title="K-Nearest Neighbor (Training data)")
    fig.update_traces(marker_size=12, marker_line_width=1.5)
    fig.update_layout(legend_orientation='h')
    return fig

def create_dtree(dtree, dimensions):
    # fig_size = plt.figure(figsize=(25,20))
    # dot_data  = tree.export_graphviz(dtree, out_file=None)
    tree.plot_tree(
        dtree,
        filled=True,
        # class_names='class',
        feature_names=dimensions
    )

def create_svm_plot(X_test, y_test, svm_predictions):
    view = PCA(n_components=2).fit(X_test)
    X_view = view.transform(X_test)

    fig = px.scatter(
        X_view, x=0, y=1,
        color=svm_predictions,
        title="SVM (Test data)")
    fig.update_traces(marker_size=12, marker_line_width=1.5)
    fig.update_layout(legend_orientation='h')
    return fig

def visualise_dtree(X_train, y_train, X_test, y_test, dec, show_train, show_test):
    # reduce dimensionality
    v = PCA(n_components=2).fit(X_train)
    Xt_train = v.transform(X_train)
    Xt_test = v.transform(X_test)

    X = np.append(Xt_train, Xt_test, axis=0)

    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])

    # evaluate decision function in a 200x200 grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # execute the decision function on the grid
    gridpts = np.array((xx, yy)).T.reshape(-1, 2)
    Z = dec(v.inverse_transform(gridpts))
    Z = Z.reshape(xx.shape).T

    # visualize decision function for these parameters
    fig = plt.figure()
    plt.pcolormesh(xx, yy, -Z)
    if show_train == "Yes":
        plt.scatter(Xt_train[:, 0], Xt_train[:, 1], c=y_train, cmap=plt.cm.RdBu_r, edgecolors="k")
    if show_test == "Yes":
        plt.scatter(Xt_test[:, 0], Xt_test[:, 1], c=y_test, cmap=plt.cm.RdBu_r, edgecolors="k")
    return fig
