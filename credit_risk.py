from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the German Credit Dataset
data = pd.read_csv("german.data", sep=r"\s+", header=None)

# Give names to the columns
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

# Display the first 5 rows
print("First 5 rows:")
print(data.head())

# Display column names
print("\nColumn names:")
print(data.columns)

# Display dataset size
print("\nDataset shape:")
print(data.shape)

# -----------------------------------
# Step 4: Understand the Dataset
# -----------------------------------

# Check basic information
print("\nDataset Information:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(data.duplicated().sum())

# Check credit risk distribution
print("\nCredit Risk Distribution:")
print(data["credit_risk"].value_counts())

# Check statistical information
print("\nStatistical Summary:")
print(data.describe())

# -----------------------------------
# Step 5: Prepare Target Variable
# -----------------------------------

# Convert credit risk into a binary risk value
# 1 = Good credit  -> 0 = Low Risk
# 2 = Bad credit   -> 1 = High Risk

data["risk"] = data["credit_risk"].map({
    1: 0,
    2: 1
})

# Display the new risk column
print("\nRisk Distribution:")
print(data["risk"].value_counts())

# Display risk labels
print("\nRisk Labels:")
print(data[["credit_risk", "risk"]].head(10))

# Separate features and target
X = data.drop(columns=["credit_risk", "risk"])
y = data["risk"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# -----------------------------------
# Step 6: Data Preprocessing
# -----------------------------------

# Identify numerical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

# Identify categorical columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("\nNumerical Features:")
print(list(numeric_features))

print("\nCategorical Features:")
print(list(categorical_features))


# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine both preprocessing methods
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

print("\nPreprocessing setup completed successfully!")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# -----------------------------------
# Step 7: Train Logistic Regression
# -----------------------------------

# Create the machine learning pipeline
logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
logistic_model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

# Make predictions
y_pred = logistic_model.predict(X_test)

print("\nPredictions:")
print(y_pred[:10])

# -----------------------------------
# Step 8: Evaluate the Model
# -----------------------------------

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print("-------------------------")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# Detailed classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Low Risk", "High Risk"]
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------------
# Step 9: Train and Evaluate Decision Tree
# -----------------------------------

tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Evaluate Decision Tree
tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_precision = precision_score(y_test, y_pred_tree)
tree_recall = recall_score(y_test, y_pred_tree)
tree_f1 = f1_score(y_test, y_pred_tree)

print("\nDecision Tree Evaluation:")
print("-------------------------")
print("Accuracy :", round(tree_accuracy, 4))
print("Precision:", round(tree_precision, 4))
print("Recall   :", round(tree_recall, 4))
print("F1 Score :", round(tree_f1, 4))

print("\nDecision Tree Classification Report:")
print(classification_report(
    y_test,
    y_pred_tree,
    target_names=["Low Risk", "High Risk"]
))

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))



# -----------------------------------
# Step 10: Random Forest Classifier
# -----------------------------------

# Create Random Forest pipeline
random_forest_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train the model
random_forest_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# Make predictions
y_pred_forest = random_forest_model.predict(X_test)

print("\nRandom Forest Predictions:")
print(y_pred_forest[:10])

# Evaluate Random Forest
forest_accuracy = accuracy_score(y_test, y_pred_forest)
forest_precision = precision_score(y_test, y_pred_forest)
forest_recall = recall_score(y_test, y_pred_forest)
forest_f1 = f1_score(y_test, y_pred_forest)

print("\nRandom Forest Evaluation:")
print("-------------------------")
print("Accuracy :", round(forest_accuracy, 4))
print("Precision:", round(forest_precision, 4))
print("Recall   :", round(forest_recall, 4))
print("F1 Score :", round(forest_f1, 4))

print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test,
    y_pred_forest,
    target_names=["Low Risk", "High Risk"]
))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_forest))

# -----------------------------------
# Step 11: Model Comparison
# -----------------------------------

# Create a comparison table
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        tree_accuracy,
        forest_accuracy
    ],
    "Precision": [
        precision,
        tree_precision,
        forest_precision
    ],
    "Recall": [
        recall,
        tree_recall,
        forest_recall
    ],
    "F1 Score": [
        f1,
        tree_f1,
        forest_f1
    ]
})

# Round the values
results[["Accuracy", "Precision", "Recall", "F1 Score"]] = \
    results[["Accuracy", "Precision", "Recall", "F1 Score"]].round(4)

print("\n========================================")
print("          MODEL COMPARISON")
print("========================================")

print(results.to_string(index=False))

# -----------------------------------
# Step 12: Feature Importance
# -----------------------------------

# Get the trained Random Forest model
forest = random_forest_model.named_steps["classifier"]

# Get the preprocessing step
preprocessor_fitted = random_forest_model.named_steps["preprocessor"]

# Get the names of the processed features
feature_names = preprocessor_fitted.get_feature_names_out()

# Get feature importance values
importance_values = forest.feature_importances_

# Create a DataFrame
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort from highest to lowest
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========================================")
print("        TOP IMPORTANT FEATURES")
print("========================================")

print(feature_importance.head(15).to_string(index=False))


# Select top 10 features
top_features = feature_importance.head(10)

# Create bar chart
plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features - Random Forest")
plt.gca().invert_yaxis()

plt.tight_layout()

# Save the graph as an image
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")

# Show the graph
plt.show()

print("\nFeature importance graph saved as feature_importance.png")