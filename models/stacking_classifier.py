from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from create_dataset_for_test import process_dataset
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from numpy import mean
from numpy import std
from matplotlib import pyplot

# APPLY THE NECESSARY CHANGES TO DATASET

# SPLIT DATA INTO TEST AND TRAIN

# evaluate a given model using cross-validation
#cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


def evaluate_model(model, X, y):
    scores = cross_val_score(
        model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# build a stacking classifier using our cross validated estimators


# get a list of models to evaluate


def get_models():
    models = dict()
    models['lr'] = LogisticRegression(max_iter=10000)
    models['knn'] = KNeighborsClassifier()
    models['r_forest'] = RandomForestClassifier(class_weight='balanced')
    models['svm'] = SVC(kernel='linear', C=1,
                        random_state=42, probability=True)
    models['bayes'] = GaussianNB()
    return models

# get a stacking ensemble of models


def get_stacking():
    # define the base models
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    level0 = list(get_models().items())
    # define meta learner model
    level1 = LogisticRegression(max_iter=10000)
    # define the stacking ensemble
    model = StackingClassifier(
        estimators=level0, final_estimator=level1, cv=cv)
    return model


def main():
    loan_dev_df, feature_cols = process_dataset("../bank_database.db")

    # CROSS_VALIDATION

    X = loan_dev_df.drop(["status", "loan_id"], axis=1)
    y = loan_dev_df["status"]
    models = get_models()
    models['stacking'] = get_stacking()

    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()


main()
