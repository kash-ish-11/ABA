import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score, accuracy_score, confusion_matrix,
    classification_report
)

st.set_page_config(page_title="ML Demonstration Tool", layout="wide")
st.title("📊 Machine Learning Demonstration Tool")
st.write("Upload any Excel/CSV file to understand Machine Learning step-by-step.")

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:

    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)

    st.subheader("🔹 Raw Data Preview")
    st.dataframe(df.head())
    st.write("Dataset Shape:", df.shape)

    # =====================================================
    # SMART COLUMN DETECTION
    # =====================================================
    cat_cols = []
    num_cols = []

    for col in df.columns:
        unique_vals = df[col].dropna().unique()

        if df[col].dtype == "object":
            cat_cols.append(col)
        elif len(unique_vals) <= 10 and set(unique_vals).issubset({0, 1}):
            cat_cols.append(col)  # binary numeric → categorical (e.g. Churn)
        else:
            num_cols.append(col)

    st.write("Categorical Columns:", cat_cols)
    st.write("Numerical Columns:", num_cols)

    # =====================================================
    # EDA
    # =====================================================
    st.subheader("🔹 Exploratory Data Analysis (EDA)")

    st.write("Missing Values (%)")
    st.dataframe((df.isnull().sum() / len(df)) * 100)

    if num_cols:
        st.subheader("Summary Statistics")
        st.dataframe(df[num_cols].describe())

    # =====================================================
    # PREPROCESSING
    # =====================================================
    st.subheader("🔹 Data Preprocessing")

    df_clean = df.copy()

    for col in df_clean.columns:
        if col in cat_cols:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
        else:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)

    st.success("Data cleaned and encoded successfully")

    scaler = StandardScaler()

    # =====================================================
    # CORRELATION ANALYSIS
    # =====================================================
    st.subheader("🔗 Correlation Analysis")

    if len(num_cols) >= 2:
        corr_matrix = df_clean[num_cols].corr()

        st.subheader("📊 Correlation Matrix")
        st.dataframe(corr_matrix.round(3))

        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            ax=ax
        )
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        st.info("""
        Interpretation:
        • Values near +1 indicate strong positive relationship
        • Values near −1 indicate strong negative relationship
        • Values near 0 indicate weak or no relationship
        """)

    # =====================================================
    # ANALYSIS TYPE
    # =====================================================
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Regression", "Classification", "Clustering"]
    )

    # =====================================================
    # REGRESSION
    # =====================================================
    if analysis_type == "Regression":

        st.subheader("📈 Linear Regression")

        target = st.selectbox("Select Target Variable (Numeric)", num_cols)

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
        st.write("R² Score:", round(r2, 3))

        if r2 < 0:
            st.warning("⚠️ Negative R² means the model performs worse than predicting the average.")

        # ---- Actual vs Predicted (Clear)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(range(len(y_test)), y_test, color="blue", label="Actual Values")
        ax.scatter(range(len(preds)), preds, color="orange", label="Predicted Values")
        ax.plot(range(len(preds)), preds, linestyle="--", color="red", label="Prediction Trend")
        ax.set_xlabel("Observation Index")
        ax.set_ylabel("Target Value")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

        # ---- Table View
        st.subheader("📋 Actual vs Predicted (Table)")
        results_df = pd.DataFrame({
            "Actual Value": y_test.values,
            "Predicted Value": preds,
            "Difference (Actual - Predicted)": y_test.values - preds
        })
        st.dataframe(results_df.head(20))

        # ---- Coefficient Table
        coef_df = pd.DataFrame({
            "Variable": X.columns,
            "Coefficient": model.coef_
        })
        coef_df["Impact"] = coef_df["Coefficient"].apply(
            lambda x: "Positive ↑" if x > 0 else "Negative ↓"
        )

        st.subheader("📊 Regression Coefficients")
        st.dataframe(coef_df.round(4))

        # ---- Regression Equation
        intercept = model.intercept_
        equation = f"{target} = {round(intercept, 3)}"
        for var, coef in zip(X.columns, model.coef_):
            sign = "+" if coef >= 0 else "-"
            equation += f" {sign} {abs(coef):.3f} × {var}"

        st.subheader("🧮 Regression Equation")
        st.code(equation)

    # =====================================================
    # CLASSIFICATION
    # =====================================================
    elif analysis_type == "Classification":

        st.subheader("📊 Classification")

        target = st.selectbox("Select Target Variable (Categorical)", cat_cols)

        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        st.subheader("📉 Target Distribution")
        st.dataframe((y.value_counts(normalize=True) * 100).rename("Percentage"))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        cm = confusion_matrix(y_test, preds)

        st.subheader("Confusion Matrix")
        st.dataframe(
            pd.DataFrame(
                cm,
                index=[f"Actual {c}" for c in np.unique(y)],
                columns=[f"Predicted {c}" for c in np.unique(y)]
            )
        )

        st.write("Accuracy:", round(accuracy_score(y_test, preds), 3))

        report = classification_report(y_test, preds, output_dict=True)
        st.subheader("📈 Classification Metrics")
        st.dataframe(pd.DataFrame(report).transpose().round(3))

    # =====================================================
    # CLUSTERING
    # =====================================================
    elif analysis_type == "Clustering":

        st.subheader("🔵 Clustering")

        k = st.slider("Select number of clusters", 2, 6, 3)

        X_scaled = scaler.fit_transform(df_clean)

        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        centroids_pca = pca.transform(kmeans.cluster_centers_)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="Set1", s=80)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="black", s=300, marker="X")

        for i, (x, y_) in enumerate(centroids_pca):
            ax.text(x, y_, f"Cluster {i}", fontsize=12, weight="bold")

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Cluster Visualization with Centroids")
        st.pyplot(fig)

        st.subheader("📊 Cluster Summary")
        st.dataframe(
            pd.DataFrame({"Cluster": clusters})
            .value_counts()
            .reset_index(name="Number of Records")
        )

    st.success("✔ Analysis completed successfully")
