from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from create_dataset_for_test import process_dataset


loan_dev_df, feature_cols = process_dataset("../bank_database.db")


X = loan_dev_df.loc[:, feature_cols]
X_trainset = X.drop(["status", "loan_id"], axis=1)
y_trainset = loan_dev_df.status

trainX, testX, trainy, testy = train_test_split(
    X_trainset, y_trainset, test_size=0.2, random_state=2, stratify=y_trainset, shuffle=True)

clf1 = LogisticRegression(random_state=1, max_iter=500)
clf2 = RandomForestClassifier(random_state=1)
clf3 = SVC(kernel='linear', C=1, random_state=42, probability=True)
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                        voting='soft'
                        )

params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [
    20, 200], 'gnb__C': [1.0, 100.0]}

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
grid = grid.fit(trainX, trainy)

print(grid)
