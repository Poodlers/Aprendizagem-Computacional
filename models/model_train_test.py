import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from create_dataset_for_test import process_dataset
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_classif, SequentialFeatureSelector
from time import time
from feature_selection import *


# APPLY THE NECESSARY CHANGES TO DATASET

# SPLIT DATA INTO TEST AND TRAIN

loan_dev_df, feature_cols = process_dataset("../bank_database.db")


X = loan_dev_df.loc[:, feature_cols]
X_trainset = X.drop(["status", "loan_id"], axis=1)
y_trainset = loan_dev_df.status


def model_train(X, y):
    # TRAIN_TEST_SPLIT
    trainX, testX, trainy, testy = train_test_split(
        X, y, test_size=0.2, random_state=2, stratify=y, shuffle=True)

    # BEST MODELS SO FAR
    # random forest classifier
    # RandomForestClassifier(class_weight='balanced')

    # Logictic Regression
    # LogisticRegression(max_iter=2000)

    # SVC
    # SVC(kernel='linear', C=1, random_state=42)

    save_results = False
    model = RandomForestClassifier(class_weight='balanced')

    model.fit(trainX, trainy)

    # Selecting features based on importance

    importance_select_features = select_features_by_importance(
        model, X, y)

    # Selecting features with Sequential Feature Selection

    # sequential_feature_select_features = sequential_select_features(
    #    model, X, y)

    model_type = model.__class__.__name__

    params = model.get_params()

    if model_type == "DecisionTreeClassifier":
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(40, 40), dpi=300)

        plot_tree(model,
                  feature_names=feature_cols,
                  class_names=["Loan given", "Loan denied"],
                  filled=True)

        fig.savefig('imagename.png')

    predict_test = True  # true in order to export a .csv for Kaggle

    Y_correct_prediction = testy
    y_predict = model.predict(testX)

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
        loan_ids.to_csv('results.csv', columns=[
                        'Id', 'Predicted'], index=False)
        print(len(test_predict_probs))

    # CALCULATE METRICS

    accuracy = accuracy_score(Y_correct_prediction, y_predict)

    mean_abs_error = mean_absolute_error(Y_correct_prediction, y_predict)

    mean_sqr_error = np.sqrt(mean_squared_error(
        Y_correct_prediction, y_predict))

    balanced_accuracy = balanced_accuracy_score(
        Y_correct_prediction, y_predict)

    recall = recall_score(Y_correct_prediction, y_predict)

    f1_sc = f1_score(Y_correct_prediction, y_predict)

    print("Accuracy: ", accuracy)
    print("Balanced Accuracy: ", balanced_accuracy)
    print("Recall: ", recall)
    print("F1 Score: ", f1_sc)
    print("Mean Absolute Error: ", mean_abs_error)
    print("Mean Squared Error: ", mean_sqr_error)

    feature_cols = list(feature_cols)

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
   # return sequential_feature_select_features, importance_select_features
    return importance_select_features


importance_select_features = model_train(
    X_trainset, y_trainset)

# print("\n \n TRAINING WITH SEQUENTIAL FEATURE VARIABLES")
# X_sequential = X_trainset.iloc[:, sequential_feature_select_features]
# model_train(X_sequential, y_trainset)

#print("\n \n TRAINING WITH IMPORTANCE FEATURE VARIABLES")
#X_importance = X_trainset.iloc[:, importance_select_features]
#model_train(X_importance, y_trainset)
