import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="ML Visualizer", layout="wide")

st.title("📊 Machine Learning Visualizer")
st.markdown("Learn ML visually with values & labels 🚀")

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Controls")

option = st.sidebar.selectbox(
    "Choose Algorithm",
   ["K-Means", "KNN", "Linear Regression", "Bar Graph"]
)

# ------------------ DATASET UPLOAD ------------------
file = st.sidebar.file_uploader(r"C:\Users\srikanth\OneDrive\Desktop\ml_visualizer\Financial_Analysis_Data-1.csv")

if file:
    data = pd.read_csv(file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

# ------------------ K-MEANS ------------------
if option == "K-Means":
    st.subheader("K-Means Clustering")
    st.write("Groups data into clusters")

    X = np.random.rand(20, 2)

    k = st.sidebar.slider("Select K", 1, 10, 3)

    model = KMeans(n_clusters=k)
    model.fit(X)

    labels = model.labels_
    centers = model.cluster_centers_

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=labels)

    # Add point labels
    for i, point in enumerate(X):
        ax.text(point[0], point[1], f"P{i}", fontsize=9)

    # Plot centers
    ax.scatter(centers[:, 0], centers[:, 1], marker='X')

    ax.set_title("K-Means Clustering")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    st.pyplot(fig)

# ------------------ KNN ------------------
elif option == "KNN":
    st.subheader("KNN Classification")
    st.write("Classifies based on nearest neighbors")

    X = np.random.rand(20, 2)
    y = np.random.randint(0, 2, 20)

    k = st.sidebar.slider("Select K", 1, 10, 3)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y)

    # Add point labels
    for i, point in enumerate(X):
        ax.text(point[0], point[1], f"P{i}", fontsize=9)

    ax.set_title("KNN Classification")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    st.pyplot(fig)

    st.write(f"Accuracy: {acc:.2f}")

# ------------------ LINEAR REGRESSION ------------------
elif option == "Linear Regression":
    st.subheader("Linear Regression")
    st.write("Fits a straight line to data")

    X = np.random.rand(20, 1)
    y = 3 * X + np.random.randn(20, 1)

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.plot(X, y_pred)

    # Add labels
    for i in range(len(X)):
        ax.text(X[i], y[i], f"P{i}", fontsize=9)

    ax.set_title("Linear Regression")
    ax.set_xlabel("X value")
    ax.set_ylabel("Y value")

    st.pyplot(fig)

elif option == "Bar Graph":
    st.subheader("Company Data Visualization")

    if data is not None:
        column = st.selectbox(
            "Select Column",
            ["Revenue", "Expenses", "Profit", "Assets", "Liabilities", "Market_Cap", "Growth_Rate", "Stock_Return"]
        )

        companies = data["Company"]
        values = data[column]

        fig, ax = plt.subplots()
        ax.bar(companies, values)

        plt.xticks(rotation=90)

        for i, v in enumerate(values):
            ax.text(i, v, str(v), ha='center', fontsize=8)

        ax.set_title(f"{column} by Company")
        ax.set_xlabel("Company")
        ax.set_ylabel(column)

        st.pyplot(fig)
    else:
        st.error("No data.csv file found")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("### 🚀 Built using Streamlit")