import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        age: int,
        monthly_income: float,
        employment_type: str,
        credit_score: float,
        existing_credit_cards: int,
        avg_monthly_spend: float,
        repayment_ratio: float,
        missed_payments_last_6m: int,
        loan_accounts: int,
        location_tier: str,
        spend_to_income_ratio: float,
        missed_payment_flag: int,
        credit_category: str,
        income_category: str,
        age_bucket: str,
    ):
        self.age = age
        self.monthly_income = monthly_income
        self.employment_type = employment_type
        self.credit_score = credit_score
        self.existing_credit_cards = existing_credit_cards
        self.avg_monthly_spend = avg_monthly_spend
        self.repayment_ratio = repayment_ratio
        self.missed_payments_last_6m = missed_payments_last_6m
        self.loan_accounts = loan_accounts
        self.location_tier = location_tier
        self.spend_to_income_ratio = spend_to_income_ratio
        self.missed_payment_flag = missed_payment_flag
        self.credit_category = credit_category
        self.income_category = income_category
        self.age_bucket = age_bucket

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "monthly_income": [self.monthly_income],
                "employment_type": [self.employment_type],
                "credit_score": [self.credit_score],
                "existing_credit_cards": [self.existing_credit_cards],
                "avg_monthly_spend": [self.avg_monthly_spend],
                "repayment_ratio": [self.repayment_ratio],
                "missed_payments_last_6m": [self.missed_payments_last_6m],
                "loan_accounts": [self.loan_accounts],
                "location_tier": [self.location_tier],
                "spend_to_income_ratio": [self.spend_to_income_ratio],
                "missed_payment_flag": [self.missed_payment_flag],
                "credit_category": [self.credit_category],
                "income_category": [self.income_category],
                "age_bucket": [self.age_bucket],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
