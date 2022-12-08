import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, roc_curve, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
from create_dataset_for_test import process_dataset
from sklearn.model_selection import cross_validate, StratifiedKFold

# APPLY THE NECESSARY CHANGES TO DATASET

# SPLIT DATA INTO TEST AND TRAIN

loan_dev_df, feature_cols = process_dataset("../bank_database.db")

X = loan_dev_df.loc[:, feature_cols]
X_trainset = X.drop(["status", "loan_id"], axis=1)
y_trainset = loan_dev_df.status


def get_stacking():
    models = dict()
    models['lr'] = LogisticRegression(max_iter=10000)
    models['knn'] = KNeighborsClassifier()
    models['r_forest'] = RandomForestClassifier(class_weight='balanced')
    models['svm'] = SVC(kernel='linear', C=1,
                        random_state=42, probability=True)
    models['bayes'] = GaussianNB()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    level0 = list(models.items())
    # define meta learner model
    level1 = LogisticRegression(max_iter=10000)
    # define the stacking ensemble

    return StackingClassifier(
        estimators=level0, final_estimator=level1, cv=cv)


# BEST MODELS SO FAR
# random forest classifier
# RandomForestClassifier(class_weight='balanced')

# GaussianNB()

# KNeighborsClassifier()

# Logictic Regression
# LogisticRegression(max_iter=2000)

# SVC
# SVC(kernel='linear', C=1, random_state=42, probability=False)

# get_stacking()


model = get_stacking()

cv = StratifiedKFold(n_splits=5, shuffle=True)


scores = cross_validate(
    model, X_trainset, y_trainset, cv=cv, scoring=('accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'recall', 'precision', 'r2'))

accuracy = np.mean(scores['test_accuracy'])
balanced_accuracy = np.mean(scores['test_balanced_accuracy'])
f1_sc = np.mean(scores['test_f1'])
recall = np.mean(scores['test_recall'])
precision = np.mean(scores['test_precision'])
auc = np.mean(scores['test_roc_auc'])
fit_times = np.mean(scores["fit_time"])

print("AUC: " + str(auc))
print("Accuracy: ", accuracy)
print("Balanced Accuracy: ", balanced_accuracy)
print("Recall: ", recall)
print("F1 Score: ", f1_sc)
print("Precision: ", precision)
print("Average Fit Times: ", fit_times)

model_type = model.__class__.__name__ + " CROSS VALIDATED"

params = model.get_params()

feature_cols = list(feature_cols)

save_results = True
if save_results:
    f = open("model_performance.txt", "a")
    f.write("\nModel Type : " + model_type + "\n")
    f.write("Params = " + params.__str__() + "\n")
    f.write("Feature cols: " + feature_cols.__str__() + "\n")
    f.write("Metrics: " + "\n")
    f.write("AUC: " + str(auc) + "\n")
    f.write("Accuracy: " + accuracy.__str__() + "\n")
    f.write("Balanced Accuracy: " + balanced_accuracy.__str__() + "\n")
    f.write("F1 Score: " + f1_sc.__str__() + "\n")
    f.write("Precision: " + precision.__str__() + "\n")
    f.write("Average Fit Times: " + fit_times.__str__() + "\n")
    f.close()
