import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# FIXED: Replaced 'model=model' with 'models' to accept your dictionary
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        # FIXED: Indented all processing lines so they stay INSIDE the for loop
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            
            # Train model
            model.fit(X_train, y_train)

            # Predict outcomes
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # FIXED: Assigned score to the correct model name mapping
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)