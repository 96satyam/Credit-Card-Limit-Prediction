from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## ROute for home page

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age = int(request.form.get('age')),
            monthly_income = float(request.form.get('monthly_income')),
            employment_type = request.form.get('employment_type'),
            credit_score = float(request.form.get('credit_score')),
            existing_credit_cards = int(request.form.get('existing_credit_cards')),
            avg_monthly_spend = float(request.form.get('avg_monthly_spend')),
            repayment_ratio = float(request.form.get('repayment_ratio')),
            missed_payments_last_6m = int(request.form.get('missed_payments_last_6m')),
            loan_accounts = int(request.form.get('loan_accounts')),
            location_tier = request.form.get('location_tier'),
            spend_to_income_ratio = float(request.form.get('spend_to_income_ratio')),
            missed_payment_flag = int(request.form.get('missed_payment_flag')),
            credit_category = request.form.get('credit_category'),
            income_category = request.form.get('income_category'),
            age_bucket = request.form.get('age_bucket')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results = results[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)