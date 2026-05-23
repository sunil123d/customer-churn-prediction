import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle

df = pd.read_excel("data/churn.xlsx")

# 🔥 STEP 1: DROP FIRST (VERY IMPORTANT)
df = df.drop(columns=[
    "CustomerID",
    "Country",
    "State",
    "City",
    "Lat Long",
    "Churn Reason",
    "Churn Label"   # 🔥 ADD THIS
])


# Convert Total Charges if needed
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

df = df.dropna()

# 🔥 STEP 2: CHECK REMAINING CATEGORICALS
print(df.select_dtypes(include="object").columns)

for col in df.select_dtypes(include="object").columns:
    print(col, df[col].nunique())

# 🔥 STEP 3: SPLIT
X = df.drop("Churn Value", axis=1)
y = df["Churn Value"]

# 🔥 STEP 4: ENCODING
X = pd.get_dummies(X, drop_first=True)

# 🔥 STEP 5: SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 STEP 6: SCALE
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("Logistic Accuracy:", accuracy_score(y_test, log_pred))
print("Logistic ROC-AUC:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:,1]))

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("RF Accuracy:", accuracy_score(y_test, rf_pred))
print("RF ROC-AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]))




pickle.dump(rf_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved successfully!")