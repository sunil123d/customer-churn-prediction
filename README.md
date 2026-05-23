# Customer Churn Prediction

End-to-end machine learning project to predict customer churn
using the IBM Telco dataset. Deployed as a Flask web application.

## Model Performance
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.84 |
| Accuracy | 81% |
| Model | Random Forest |
| Features | 36 |
| Dataset | IBM Telco (7043 customers) |

## Tech Stack
- Python, Flask, scikit-learn, pandas, pickle

## How to Run
1. Clone the repo
   git clone https://github.com/sunil123d/customer-churn-prediction.git

2. Install dependencies
   pip install -r requirements.txt

3. Add dataset
   Place churn.xlsx inside data/ folder

4. Train the model
   python train.py

5. Run the app
   python app.py

6. Open browser
   http://localhost:5000
