# =====================================
# 1. IMPORT LIBRARIES
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# =====================================
# 2. LOAD DATA
# =====================================
df = pd.read_csv("data/spotify_data.csv")
df.columns = df.columns.str.strip()

print("Dataset Loaded")
print(df.head())


# =====================================
# 3. CREATE CHURN LABEL
# =====================================
df["churn"] = np.where(
    (df["premium_sub_willingness"] == "No") &
    (df["music_recc_rating"] <= 3),
    1,
    0
)

print("\nChurn Distribution:")
print(df["churn"].value_counts())

if df["churn"].nunique() < 2:
    raise ValueError("Churn column has only one class.")


# =====================================
# 4. VISUALIZE CHURN DISTRIBUTION (MATPLOTLIB)
plt.figure()
df["churn"].value_counts().plot(kind="bar")
plt.xticks([0, 1], ["No Churn", "Churn"], rotation=0)
plt.xlabel("Churn Status")
plt.ylabel("Number of Users")
plt.title("Churn Distribution")
plt.show()


# =====================================
# 5. FEATURE SELECTION
# =====================================
features = [
    "Age",
    "Gender",
    "spotify_usage_period",
    "spotify_listening_device",
    "spotify_subscription_plan",
    "premium_sub_willingness",
    "preferred_listening_content",
    "fav_music_genre",
    "music_time_slot",
    "music_lis_frequency",
    "music_recc_rating"
]

X = df[features]
y = df["churn"]

X = X.fillna("Unknown")
X = pd.get_dummies(X, drop_first=True)


# =====================================
# 6. TRAIN-TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =====================================
# 7. FEATURE SCALING
# =====================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =====================================
# 8. TRAIN PREDICTIVE MODEL
# =====================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# =====================================
# 9. MODEL EVALUATION
# =====================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))


# =====================================
# 10. CONFUSION MATRIX (MATPLOTLIB)
# =====================================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()


# =====================================
# 11. ROC CURVE (MATPLOTLIB)
# =====================================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# =====================================
# 12. PREDICT CHURN PROBABILITY (FULL DATA)
# =====================================
df["churn_probability"] = model.predict_proba(
    scaler.transform(X)
)[:, 1]

df["predicted_churn"] = np.where(df["churn_probability"] >= 0.5, 1, 0)

print("\nPrediction Sample:")
print(df[["churn_probability", "predicted_churn"]].head())


# =====================================
# 13. CHURN RISK CHUNKS (SEGMENTS)
# =====================================
def churn_segment(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

df["churn_segment"] = df["churn_probability"].apply(churn_segment)

print("\nChurn Segments:")
print(df["churn_segment"].value_counts())


# =====================================
# 14. VISUALIZE CHURN SEGMENTS (MATPLOTLIB)
# =====================================
plt.figure()
df["churn_segment"].value_counts().plot(kind="bar")
plt.xlabel("Churn Segment")
plt.ylabel("Number of Users")
plt.title("Customer Churn Risk Segments")
plt.show()


# =====================================
# 15. SAVE MODEL
# =====================================
joblib.dump(model, "spotify_churn_model.pkl")
joblib.dump(scaler, "spotify_scaler.pkl")

print("\nModel & scaler saved successfully")
