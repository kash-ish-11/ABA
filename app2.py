import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

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

st.write("""
This tool demonstrates **Machine Learning concepts step-by-step** using real business data.
It is designed for **easy understanding by management and non-technical users**.
""")

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

st.info("""
📂 **Dataset Upload**
- Upload any Excel or CSV file.
- The system automatically identifies variable types.
- No prior data preparation is required.
""")

if uploaded_file:

    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)

    st.subheader("🔹 Raw Data Preview")
    st.dataframe(df.head())
    st.write("Dataset Shape:", df.shape)

    # =====================================================
    # SMART COLUMN DETECTION
    # =====================================================
    cat_cols, num_cols = [], []

    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if df[col].dtype == "object":
            cat_cols.append(col)
        elif len(unique_vals) <= 10 and set(unique_vals).issubset({0, 1}):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    st.write("Categorical Columns:", cat_cols)
    st.write("Numerical Columns:", num_cols)

    st.info("""
🧠 **Automatic Variable Detection**
- Categorical variables → Classification
- Numerical variables → Regression & Clustering
- Binary (0/1) variables → Classification targets
""")

    # =====================================================
    # EDA
    # =====================================================
    st.subheader("🔹 Exploratory Data Analysis (EDA)")
    st.dataframe((df.isnull().sum() / len(df)) * 100)

    if num_cols:
        st.dataframe(df[num_cols].describe())

    st.info("""
🔍 **EDA Purpose**
- Identifies missing values and data quality issues
- Helps understand scale, spread, and distributions
- Foundation for reliable modeling
""")

    # =====================================================
    # PREPROCESSING
    # =====================================================
    df_clean = df.copy()

    for col in df_clean.columns:
        if col in cat_cols:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
        else:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)

    scaler = StandardScaler()

    st.info("""
🛠 **Data Preprocessing**
- Missing values handled automatically
- Categorical values encoded numerically
- Numerical variables standardized where required
""")

    # =====================================================
    # CORRELATION
    # =====================================================
    if len(num_cols) >= 2:
        corr = df_clean[num_cols].corr()
        st.dataframe(corr.round(3))

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

        st.info("""
🔗 **Correlation Analysis**
- Measures relationship between numerical variables
- Helps detect multicollinearity
- Important before regression modeling
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

        st.info("""
📈 **Regression Objective**
- Predicts a numerical target variable
- Explains how input variables impact the outcome
""")

        confidence_level = st.selectbox(
            "Select Confidence Level",
            ["90%", "95%", "99%"],
            index=1
        )

        alpha_map = {"90%": 0.10, "95%": 0.05, "99%": 0.01}
        alpha = alpha_map[confidence_level]

        st.info("""
📐 **Confidence Level**
- α = 1 − Confidence Level
- Used for:
  • Overall model significance (ANOVA)
  • Individual variable significance (p-values)
""")

        target = st.selectbox("Select Target Variable (Numeric)", num_cols)

        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        r2 = r2_score(y_test, preds)

        st.subheader("📐 Model Fit Metrics")
        st.dataframe(pd.DataFrame({
            "Metric": ["R²", "Absolute R²"],
            "Value": [round(r2, 4), round(abs(r2), 4)]
        }))

        st.info("""
📊 **R² Interpretation**
- Measures how much variation is explained by the model
- Absolute R² avoids confusion if R² is negative
- R² does NOT depend on confidence level
""")

        # Actual vs Predicted
        st.subheader("📋 Actual vs Predicted")

        avp_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": preds,
            "Residual": y_test.values - preds
        })
        st.dataframe(avp_df.head(20))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(range(len(y_test)), y_test, label="Actual")
        ax.scatter(range(len(preds)), preds, label="Predicted")
        ax.plot(range(len(preds)), preds, linestyle="--", color="red")
        ax.legend()
        st.pyplot(fig)

        st.info("""
📋 **Actual vs Predicted**
- Shows real vs model-generated values
- Residual indicates prediction error
""")

        # Coefficients
        coef_df = pd.DataFrame({
            "Variable": X.columns,
            "Coefficient": model.coef_
        })
        st.dataframe(coef_df.round(4))

        st.info("""
📊 **Coefficients**
- Positive → direct relationship
- Negative → inverse relationship
""")

        # P-values
        X_sm = sm.add_constant(X)
        sm_model = sm.OLS(y, X_sm).fit()

        pval_df = pd.DataFrame({
            "Variable": sm_model.params.index,
            "Coefficient": sm_model.params.values,
            "P-value": sm_model.pvalues.values,
            "Include (p < α)": ["Yes" if p < alpha else "No" for p in sm_model.pvalues]
        })

        st.subheader("📊 Coefficient Significance (P-values)")
        st.dataframe(pval_df.round(6))

        st.info("""
📉 **P-value Decision Rule**
- p < α → variable is statistically significant
- Only significant variables should be used for interpretation
""")

        # Regression Equation
        significant_terms = [
            f"{c:.3f} × {v}" for v, c, p in zip(
                pval_df["Variable"], pval_df["Coefficient"], pval_df["P-value"]
            ) if v != "const" and p < alpha
        ]

        equation = f"{target} = " + " + ".join(significant_terms) if significant_terms else \
            "No statistically significant predictors"

        st.subheader("🧮 Regression Equation (Significant Variables Only)")
        st.code(equation)

        # ANOVA
        y_hat = sm_model.predict(X_sm)
        y_mean = np.mean(y)

        ss_total = np.sum((y - y_mean) ** 2)
        ss_reg = np.sum((y_hat - y_mean) ** 2)
        ss_res = np.sum((y - y_hat) ** 2)

        n, k = len(y), X.shape[1]
        ms_reg = ss_reg / k
        ms_res = ss_res / (n - k - 1)

        f_stat = ms_reg / ms_res
        significance_f = 1 - stats.f.cdf(f_stat, k, n - k - 1)

        anova_df = pd.DataFrame({
            "Source": ["Regression", "Residual", "Total"],
            "df": [k, n - k - 1, n - 1],
            "SS": [ss_reg, ss_res, ss_total],
            "MS": [ms_reg, ms_res, ""],
            "F": [f_stat, "", ""],
            "Significance F": [significance_f, "", ""]
        })

        st.subheader("📊 ANOVA – Overall Model Significance")
        st.dataframe(anova_df.round(6))

        st.info("""
📊 **ANOVA Interpretation**
- Tests whether the model is statistically significant overall
- If Significance F < α → model is significant
""")

    # =====================================================
    # CLASSIFICATION
    # =====================================================
    elif analysis_type == "Classification":

        st.subheader("📊 Classification")

        st.info("""
📊 **Classification Objective**
- Predicts categorical outcomes (e.g., Churn / No Churn)
""")

        target = st.selectbox("Select Target Variable (Categorical)", cat_cols)

        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        cm = confusion_matrix(y_test, preds)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {cls}" for cls in np.unique(y)],
            columns=[f"Predicted {cls}" for cls in np.unique(y)]
        )

        st.subheader("📊 Confusion Matrix (Actual vs Predicted)")
        st.dataframe(cm_df)

        st.write("Accuracy:", accuracy_score(y_test, preds))
        st.dataframe(pd.DataFrame(classification_report(y_test, preds, output_dict=True)).T)

        st.info("""
📈 **Classification Metrics**
- Precision: Correctness of positive predictions
- Recall: Ability to detect positives
- F1-score: Balance between precision & recall
""")

    # =====================================================
    # CLUSTERING
    # =====================================================
    elif analysis_type == "Clustering":

        st.subheader("🔵 Clustering")

        st.info("""
🔵 **Clustering Objective**
- Groups similar data points
- No target variable is used
""")

        k = st.slider("Select number of clusters", 2, 6, 3)

        X_scaled = scaler.fit_transform(df_clean)

        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        centers = pca.transform(kmeans.cluster_centers_)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="Set1")
        ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=200)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)

        st.dataframe(pd.DataFrame({
            "Component": ["PC1", "PC2"],
            "Variance Explained (%)": pca.explained_variance_ratio_ * 100
        }).round(2))

        st.info("""
🔍 **Clustering Explanation**
- Clusters are formed using all standardized variables
- PCA is used only for visualization
- PC1 & PC2 summarize original variables
""")
