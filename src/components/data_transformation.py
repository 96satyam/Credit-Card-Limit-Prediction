import sys
import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_columns = [
                'age',
                'monthly_income',
                'credit_score',
                'existing_credit_cards',
                'avg_monthly_spend',
                'repayment_ratio',
                'missed_payments_last_6m',
                'loan_accounts',
                'spend_to_income_ratio',
                'missed_payment_flag'
            ]

            categorical_columns = [
                'employment_type',
                'location_tier',
                'credit_category',
                'income_category',
                'age_bucket'
            ]
            
            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Numerical and Categorical pipeline created successfully.")

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data read successfully.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "recommended_credit_limit"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing pipeline on train and test features.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving the preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
