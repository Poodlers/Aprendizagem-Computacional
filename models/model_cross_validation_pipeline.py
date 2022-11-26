from numpy import NaN, sqrt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import sqlite3
from datetime import datetime
from encoder_one_hot import CategoricalOneHot
import matplotlib.pyplot as plt
from create_dataset_for_test import process_dataset
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from stacking_classifier import get_stacking

# APPLY THE NECESSARY CHANGES TO DATASET

# SPLIT DATA INTO TEST AND TRAIN

loan_dev_df, feature_cols = process_dataset("../bank_database.db")

# BEST MODELS SO FAR
# random forest classifier
# RandomForestClassifier(class_weight='balanced')

# Logictic Regression
# LogisticRegression(max_iter=2000)

# SVC
# SVC(kernel='linear', C=1, random_state=42, probability=False)


model = get_stacking()

model_type = model.__class__.__name__

params = model.get_params()

# CROSS_VALIDATION

X = loan_dev_df.drop(["status", "loan_id"], axis=1)
y = loan_dev_df["status"]

# evaluate a given model using cross-validation


Y_correct_prediction = y

predict_test = False  # true in order to export a .csv for Kaggle

if predict_test:
    loan_test_df, feature_test_cols = process_dataset(
        "test_data/test_database.db")

    loan_ids = pd.DataFrame(loan_test_df["loan_id"])
    X_test_predict = loan_test_df.drop(["status", "loan_id"], axis=1)
    X_test_predict = X_test_predict.dropna()
    test_predict_probs = pd.DataFrame(model.predict_proba(X_test_predict))
    print(model.classes_)
    print(test_predict_probs)
    loan_ids = loan_ids.assign(Predicted=test_predict_probs[0])
    loan_ids = loan_ids.rename(columns={'loan_id': 'Id'})
    print(loan_ids)
    loan_ids.to_csv('results.csv', columns=['Id', 'Predicted'], index=False)
    print(len(test_predict_probs))

# CALCULATE METRICS

accuracy = accuracy_score(Y_correct_prediction, y_predict)

mean_abs_error = mean_absolute_error(Y_correct_prediction, y_predict)

mean_sqr_error = sqrt(mean_squared_error(Y_correct_prediction, y_predict))

balanced_accuracy = balanced_accuracy_score(Y_correct_prediction, y_predict)

recall = recall_score(Y_correct_prediction, y_predict)

f1_sc = f1_score(Y_correct_prediction, y_predict)

print("Accuracy: ", accuracy)
print("Balanced Accuracy: ", balanced_accuracy)
print("Recall: ", recall)
print("F1 Score: ", f1_sc)
print("Mean Absolute Error: ", mean_abs_error)
print("Mean Squared Error: ", mean_sqr_error)

feature_cols = list(feature_cols)

save_results = True
if save_results:
    f = open("model_performance.txt", "a")
    f.write("\nModel Type : " + model_type + "\n")
    f.write("Params = " + params.__str__() + "\n")
    f.write("Feature cols: " + feature_cols.__str__() + "\n")
    f.write("Metrics: " + "\n")
    f.write("Accuracy: " + accuracy.__str__() + "\n")
    f.write("Balanced Accuracy: " + balanced_accuracy.__str__() + "\n")
    f.write("Recall: " + recall.__str__() + "\n")
    f.write("F1 Score: " + f1_sc.__str__() + "\n")
    f.write("Mean Absolute Error: " + mean_abs_error.__str__() + "\n")
    f.write("Mean Squared Error: " + mean_sqr_error.__str__() + "\n")

    f.close()
