import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "Upload a credit-card transaction CSV file to explore the data "
    "and train a model to classify transactions as legitimate or fraudulent."
)

uploaded_file = st.file_uploader("Upload creditcard CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload your CSV dataset to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if "Class" not in df.columns:
    st.error("The dataset must contain a 'Class' column (0 = Non-Fraud, 1 = Fraud).")
    st.stop()

st.header("1. Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Missing values", int(df.isna().sum().sum()))

with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Dataset information"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(dtype) for dtype in df.dtypes],
        "Missing values": [int(value) for value in df.isna().sum()]
    })
    st.dataframe(info_df, use_container_width=True)
    st.write("Summary statistics")
    st.dataframe(df.describe().T, use_container_width=True)

st.header("2. Exploratory data analysis")

class_counts = df["Class"].value_counts().sort_index()
class_names = {0: "Non-Fraud", 1: "Fraud"}
distribution = pd.DataFrame({
    "Transaction type": [class_names.get(index, str(index)) for index in class_counts.index],
    "Count": class_counts.values
})

left, right = st.columns(2)
with left:
    st.subheader("Class distribution")
    fig, ax = plt.subplots()
    ax.bar(distribution["Transaction type"], distribution["Count"])
    ax.set_ylabel("Number of transactions")
    ax.set_xlabel("Class")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Transaction amount")
    if "Amount" in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="Amount", bins=40, ax=ax)
        ax.set_title("Distribution of transaction amount")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("The dataset does not contain an Amount column.")

with st.expander("Correlation matrix"):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

st.header("3. Train fraud detection model")

model_choice = st.selectbox(
    "Choose a machine-learning model",
    ["Logistic Regression", "XGBoost"]
)

test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)

if st.button("Train model", type="primary"):
    with st.spinner("Training model. Please wait..."):
        try:
            clean_df = df.dropna().copy()
            X = clean_df.drop(columns=["Class"])
            y = clean_df["Class"]

            if y.nunique() != 2:
                st.error("The Class column must contain two classes: 0 and 1.")
                st.stop()

            if not set(y.unique()).issubset({0, 1}):
                st.error("Please encode Class as 0 = Non-Fraud and 1 = Fraud.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size / 100,
                random_state=42,
                stratify=y
            )

            # Apply SMOTE only to the training data, never to the test data.
            if y_train.value_counts().min() < 2:
                st.error("There are too few minority-class examples to apply SMOTE.")
                st.stop()

            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.1,
                    max_depth=4,
                    min_child_weight=3,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    gamma=1,
                    reg_alpha=2,
                    reg_lambda=2,
                    eval_metric="logloss",
                    random_state=42
                )

            model.fit(X_train_resampled, y_train_resampled)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)

            st.success("Model training completed!")

            st.subheader("Model performance on test data")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{accuracy:.4f}")
            m2.metric("Precision", f"{precision:.4f}")
            m3.metric("Recall", f"{recall:.4f}")

            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_test, predictions, labels=[0, 1])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Non-Fraud", "Actual Fraud"],
                columns=["Predicted Non-Fraud", "Predicted Fraud"]
            )
            st.dataframe(cm_df, use_container_width=True)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Non-Fraud", "Fraud"],
                        yticklabels=["Non-Fraud", "Fraud"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Classification report")
            report = classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        except Exception as exc:
            st.error(f"An error occurred while training the model: {exc}")

st.caption("Note: This is an educational mini-project. It is not a real banking fraud prevention system.")
