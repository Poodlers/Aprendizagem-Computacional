from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from create_dataset_for_test import process_dataset


loan_dev_df, feature_cols = process_dataset("../bank_database.db")


X = loan_dev_df.loc[:, feature_cols]
X_trainset = X.drop(["status", "loan_id"], axis=1)
y_trainset = loan_dev_df.status

trainX, testX, trainy, testy = train_test_split(
    X_trainset, y_trainset, test_size=0.2, random_state=2, stratify=y_trainset, shuffle=True)

clf1 = LogisticRegression(random_state=1, max_iter=10000)
clf2 = RandomForestClassifier(class_weight='balanced')
# eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)],
#                        voting='soft'
#                        )

params = {'n_estimators': [
    20, 200, 300, 400, 500], 'criterion': ["gini", "entropy", "log_loss"]}

cv = StratifiedKFold(n_splits=5, shuffle=True)
grid = GridSearchCV(estimator=clf2, param_grid=params, cv=cv)
grid = grid.fit(trainX, trainy)

print(" Results from Grid Search ")
print("\n The best estimator across ALL searched params:\n", grid.best_estimator_)
print("\n The best score across ALL searched params:\n", grid.best_score_)
print("\n The best parameters across ALL searched params:\n", grid.best_params_)
