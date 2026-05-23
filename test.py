import pandas as pd
import pickle

df = pd.read_excel("data/churn.xlsx")   # OR churn.csv
print(df.head())


pickle.dump(rf_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved successfully!")