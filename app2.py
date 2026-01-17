#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix


st.set_page_config(page_title="Management-Friendly ML Prototype", layout="wide")

st.title("📊 Machine Learning Prototype for Management Understanding")
st.write("This tool demonstrates **how business data is treated step-by-step using Machine Learning**.")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("🔹 Raw Data Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # DATA PREPROCESSING
    # --------------------------------------------------
    df_clean = df.copy()

    # Remove ID-like columns
    for col in df_clean.columns:
        if "id" in col.lower():
            df_clean.drop(columns=[col], inplace=True)

    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].dtype in ["int64", "float64"]:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    # Encode categorical variables
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

    # --------------------------------------------------
    # MACHINE SUGGESTS TARGETS
    # --------------------------------------------------
    regression_targets = []
    classification_targets = []

    for col in df_clean.columns:
        unique_vals = df_clean[col].nunique()
        if unique_vals > 10:
            regression_targets.append(col)
        elif unique_vals <= 5:
            classification_targets.append(col)

    st.subheader("🔹 Machine Suggestions")
    st.write("**Possible Regression Targets:**", regression_targets)
    st.write("**Possible Classification Targets:**", classification_targets)

    # --------------------------------------------------
    # USER DECISION
    # --------------------------------------------------
    st.subheader("🔹 Choose Analysis Type")

    analysis_type = st.selectbox(
        "Select what you want to perform:",
        ["Regression", "Classification", "Unsupervised Learning", "Conjoint Analysis"]
    )

    scaler = StandardScaler()

    # --------------------------------------------------
    # REGRESSION
    # --------------------------------------------------
    if analysis_type == "Regression":
        if not regression_targets:
            st.error("No suitable regression target found.")
        else:
            target = st.selectbox("Select Target Variable", regression_targets)

            X = df_clean.drop(columns=[target])
            y = df_clean[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = LinearRegression()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)

            st.subheader("📈 Regression Results")
            st.write("**R² Score:**", round(r2, 3))

            fig, ax = plt.subplots()
            ax.scatter(y_test, preds)
            ax.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()], color="red")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

            equation = f"{target} = {model.intercept_:.2f}"
            for coef, col in zip(model.coef_, X.columns):
                equation += f" + ({coef:.2f} × {col})"

            st.write("**Regression Equation:**")
            st.code(equation)

    # --------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------
    elif analysis_type == "Classification":
        if not classification_targets:
            st.error("No suitable classification target found.")
        else:
            target = st.selectbox("Select Target Variable", classification_targets)

            X = df_clean.drop(columns=[target])
            y = df_clean[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)

            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)

            st.subheader("📊 Classification Results")
            st.write("**Accuracy:**", round(acc, 3))
            st.write("**Confusion Matrix:**")
            st.dataframe(confusion_matrix(y_test, preds))

    # --------------------------------------------------
    # UNSUPERVISED LEARNING
    # --------------------------------------------------
    elif analysis_type == "Unsupervised Learning":
        k = st.slider("Select Number of Clusters (k)", 2, 10, 3)

        scaled_data = scaler.fit_transform(df_clean)

        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        df_clustered = df_clean.copy()
        df_clustered["Cluster"] = clusters

        st.subheader("📌 Cluster Summary")
        st.dataframe(df_clustered["Cluster"].value_counts().sort_index())

        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)

        fig, ax = plt.subplots()
        ax.scatter(components[:,0], components[:,1], c=clusters)
        ax.set_title("PCA Visualization of Clusters")
        st.pyplot(fig)

    # --------------------------------------------------
    # CONJOINT ANALYSIS
    # --------------------------------------------------
    elif analysis_type == "Conjoint Analysis":
        if not regression_targets:
            st.error("No suitable preference / rating variable found.")
        else:
            target = st.selectbox("Select Preference / Rating Variable", regression_targets)

            X = df_clean.drop(columns=[target])
            y = df_clean[target]

            model = LinearRegression()
            model.fit(X, y)

            conjoint_table = pd.DataFrame({
                "Attribute": X.columns,
                "Part-Worth Utility": model.coef_
            }).sort_values(by="Part-Worth Utility", ascending=False)

            st.subheader("🎯 Conjoint Analysis Results")
            st.dataframe(conjoint_table)

            st.info("""
            **Interpretation**
            • Positive values → Increase preference  
            • Negative values → Decrease preference  
            • Larger magnitude → Higher importance  
            """)

    st.success("✔ Analysis completed based on your selection")


# In[ ]:




