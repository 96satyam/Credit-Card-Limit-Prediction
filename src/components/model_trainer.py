import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    

)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor 

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config =ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose = False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
    "Random Forest": {
        "n_estimators": [100],
        "max_depth": [None],
        "min_samples_split": [2],
    },
    "Decision Tree": {
        "max_depth": [None],
        "min_samples_split": [2],
    },
    "Gradient Boosting": {
        "n_estimators": [100],
        "learning_rate": [0.1],
        "max_depth": [3],
    },
    "Linear Regression": {
        # No hyperparameters to tune normally, leave empty
    },
    "K-Neighbors Classifier": {
        "n_neighbors": [5],
        "weights": ["uniform"],
    },
    "XGBRegressor": {
        "n_estimators": [100],
        "learning_rate": [0.1],
        "max_depth": [3],
    },
    "CatBoosting Regressor": {
        "iterations": [100],
        "learning_rate": [0.1],
        "depth": [3],
    },
    "AdaBoost Regressor": {
        "n_estimators": [50],
        "learning_rate": [1.0],
    },
}



            model_report: dict = evaluate_models(X_train =X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params =params )

            # Best model score

            best_model_score = max(sorted(model_report.values()))

            # Best model name

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset {best_model_name}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square


        except Exception as e:
            raise CustomException(e, sys)
            