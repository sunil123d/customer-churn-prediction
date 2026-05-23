import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("data/churn.xlsx")

# 🔥 STEP 1: DROP FIRST (VERY IMPORTANT)
df = df.drop(columns=[
    "CustomerID",
    "Country",
    "State",
    "City",
    "Lat Long",
    "Churn Reason"
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