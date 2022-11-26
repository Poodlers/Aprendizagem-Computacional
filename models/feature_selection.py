import matplotlib.pyplot as plt
from time import time
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_classif, SequentialFeatureSelector


def select_features_by_importance(model, X, y):
    # Selecting features based on importance
    importance = None
    if model.__class__.__name__ == 'RandomForestClassifier':
        importance = np.abs(model.feature_importances_)
    else:
        importance = np.abs(model.coef_[0])
    feature_names = np.array(X.columns)
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(20, 10))
    plt.bar(height=importance, x=feature_names)
    plt.title("Feature importances via coefficients")
    plt.show()

    threshold = np.sort(importance)[-8]

    tic = time()
    sfm = SelectFromModel(model, max_features=10).fit(X, y)
    toc = time()
    print(
        f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
    print(f"Done in {toc - tic:.3f}s")
    return feature_names[sfm.get_support()]


def sequential_select_features(model, X, y):
    feature_names = np.array(X.columns)
    tic_fwd = time()
    sfs_forward = SequentialFeatureSelector(
        model, n_features_to_select=10, direction="forward"
    ).fit(X, y)
    toc_fwd = time()

    print(
        "Features selected by forward sequential selection: "
        f"{feature_names[sfs_forward.get_support()]}"
    )
    print(f"Done in {toc_fwd - tic_fwd:.3f}s")
    return feature_names[sfs_forward.get_support()]


def select_k_best(X, y):
    # Univariate feature_selection
    # Create and fit selector
    selector = SelectKBest(f_classif, k=7)
    selector.fit(X, y)
    # Get columns to keep and create new dataframe with those only
    cols = selector.get_support(indices=True)
    X_new = X.iloc[:, cols]
    print("Cols returned by f_classif method: ", X_new.columns)
