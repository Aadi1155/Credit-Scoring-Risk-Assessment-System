import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="💳",
    layout="wide"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("💳 Credit Scoring & Risk Assessment System")

st.write(
    "This system uses a Machine Learning model to assess "
    "credit risk based on applicant information."
)


# -----------------------------------
# Load Dataset
# -----------------------------------

data = pd.read_csv(
    "german.data",
    sep=r"\s+",
    header=None
)


# -----------------------------------
# Column Names
# -----------------------------------

columns = [
    "checking_account",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "credit_risk"
]

data.columns = columns


# -----------------------------------
# Target Variable
# -----------------------------------

data["risk"] = data["credit_risk"].map({
    1: 0,
    2: 1
})

X = data.drop(columns=["credit_risk", "risk"])
y = data["risk"]


# -----------------------------------
# Preprocessing
# -----------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# -----------------------------------
# Train Model
# -----------------------------------

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])


model.fit(X, y)
# Model Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)
# Feature Importance
forest = model.named_steps["classifier"]
preprocessor_fitted = model.named_steps["preprocessor"]

feature_names = preprocessor_fitted.get_feature_names_out()
importance_values = forest.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

y_test_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)


# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.header("Applicant Information")


duration = st.sidebar.number_input(
    "Loan Duration (months)",
    min_value=1,
    max_value=72,
    value=12
)


credit_amount = st.sidebar.number_input(
    "Credit Amount",
    min_value=100,
    max_value=20000,
    value=2500
)


age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)


installment_rate = st.sidebar.selectbox(
    "Installment Rate",
    [1, 2, 3, 4],
    index=1
)


residence_since = st.sidebar.selectbox(
    "Residence Since",
    [1, 2, 3, 4],
    index=2
)


existing_credits = st.sidebar.selectbox(
    "Existing Credits",
    [1, 2, 3, 4],
    index=0
)


dependents = st.sidebar.selectbox(
    "Number of Dependents",
    [1, 2],
    index=0
)


# -----------------------------------
# Categorical Inputs
# -----------------------------------

checking_account = st.sidebar.selectbox(
    "Checking Account",
    sorted(data["checking_account"].unique())
)

credit_history = st.sidebar.selectbox(
    "Credit History",
    sorted(data["credit_history"].unique())
)

purpose = st.sidebar.selectbox(
    "Purpose",
    sorted(data["purpose"].unique())
)

savings_account = st.sidebar.selectbox(
    "Savings Account",
    sorted(data["savings_account"].unique())
)

employment = st.sidebar.selectbox(
    "Employment",
    sorted(data["employment"].unique())
)

personal_status_sex = st.sidebar.selectbox(
    "Personal Status & Sex",
    sorted(data["personal_status_sex"].unique())
)

other_debtors = st.sidebar.selectbox(
    "Other Debtors",
    sorted(data["other_debtors"].unique())
)

property_value = st.sidebar.selectbox(
    "Property",
    sorted(data["property"].unique())
)

other_installment_plans = st.sidebar.selectbox(
    "Other Installment Plans",
    sorted(data["other_installment_plans"].unique())
)

housing = st.sidebar.selectbox(
    "Housing",
    sorted(data["housing"].unique())
)

job = st.sidebar.selectbox(
    "Job",
    sorted(data["job"].unique())
)

telephone = st.sidebar.selectbox(
    "Telephone",
    sorted(data["telephone"].unique())
)

foreign_worker = st.sidebar.selectbox(
    "Foreign Worker",
    sorted(data["foreign_worker"].unique())
)


# -----------------------------------
# Prediction Button
# -----------------------------------

if st.button("🔍 Predict Credit Risk"):

    applicant = pd.DataFrame([{
        "checking_account": checking_account,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_account": savings_account,
        "employment": employment,
        "installment_rate": installment_rate,
        "personal_status_sex": personal_status_sex,
        "other_debtors": other_debtors,
        "residence_since": residence_since,
        "property": property_value,
        "age": age,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "dependents": dependents,
        "telephone": telephone,
        "foreign_worker": foreign_worker
    }])


    prediction = model.predict(applicant)[0]

    probability = model.predict_proba(applicant)[0]


    # -----------------------------------
    # Display Result
    # -----------------------------------

    st.subheader("Credit Risk Result")


    if prediction == 1:

        st.error("⚠️ HIGH RISK")

        st.write(
            f"High Risk Probability: "
            f"{probability[1] * 100:.2f}%"
        )

    else:

        st.success("✅ LOW RISK")

        st.write(
            f"Low Risk Probability: "
            f"{probability[0] * 100:.2f}%"
        )


# -----------------------------------
# About Section
# -----------------------------------

st.divider()
# Feature Importance
st.subheader("⭐ Important Factors in Risk Prediction")

forest = model.named_steps["classifier"]
preprocessor_fitted = model.named_steps["preprocessor"]

feature_names = preprocessor_fitted.get_feature_names_out()
importance_values = forest.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

top_features = feature_importance.head(10)

st.dataframe(
    top_features,
    use_container_width=True
)
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Precision", f"{precision * 100:.2f}%")
col3.metric("Recall", f"{recall * 100:.2f}%")
col4.metric("F1 Score", f"{f1 * 100:.2f}%")

st.subheader("🔲 Confusion Matrix")

st.write("Rows = Actual Risk | Columns = Predicted Risk")

cm_df = pd.DataFrame(
    cm,
    index=["Actual Low Risk", "Actual High Risk"],
    columns=["Predicted Low Risk", "Predicted High Risk"]
)

st.dataframe(cm_df)


st.subheader("About This Project")

st.write(
    "This project uses the German Credit Dataset and a "
    "Random Forest Classification model to demonstrate "
    "credit risk assessment using Machine Learning."
)

st.info(
    "This is an educational project and should not be used "
    "as the sole basis for real financial lending decisions."
)