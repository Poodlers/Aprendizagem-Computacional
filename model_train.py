from numpy import NaN, var
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, balanced_accuracy_score
import pandas as pd
import sqlite3
from datetime import datetime

conn = sqlite3.connect('bank_database.db')

loan_dev_df = pd.read_sql('''SELECT * FROM account
     JOIN loan_dev ON loan_dev.account_id = account.account_id
    LEFT JOIN disp ON disp.account_id = account.account_id
    LEFT JOIN client ON client.client_id = disp.client_id
    LEFT JOIN district ON district.code = client.district_id
    WHERE disp.type = "OWNER"
''', conn)


loan_dev_df = loan_dev_df.replace('?', NaN)
loan_dev_df = loan_dev_df.loc[:, ~loan_dev_df.columns.duplicated()]


age_dict = {"M": 0, "F": 1}
frequency_dict = {"monthly issuance": 0,
                  "issuance after transaction": 1, "weekly issuance": 2}


def make_into_discrete(var1):
    return age_dict.get(var1)


def make_frequency_discrete(var1):
    return frequency_dict.get(var1)

# create age_at_loan


train, test = train_test_split(loan_dev_df, test_size=0.2, random_state=0)


age_at_loan = []
for i in range(train['birth_number'].size):
    try:
        train['date'][i]
    except:
        age_at_loan.append(NaN)
        continue

    age_at_loan.append((datetime.strptime(train['date'][i], '%Y-%m-%d').date(
    ) - datetime.strptime(train['birth_number'][i], '%Y-%m-%d').date()).days / 365.25)

train = train.assign(age_at_loan=age_at_loan)

train['gender'] = train['gender'].apply(make_into_discrete)
train['frequency'] = train['frequency'].apply(
    make_frequency_discrete)

train = train.dropna()


feature_cols = train.columns.drop(
    ["date", "type", "name", "region", "birth_number", "no. of municipalities with inhabitants < 499 ",
     "no. of municipalities with inhabitants 500-1999", "no. of municipalities with inhabitants 2000-9999 ",
     "no. of municipalities with inhabitants >10000 ", "loan_id", "district_id", "account_id", "status"
     ])

print(feature_cols)


X = train.loc[:, feature_cols]

y = train.status

# 2. instantiate model
logreg = LogisticRegression(
    solver='liblinear', max_iter=5000, class_weight='balanced')

# 3. fit
logreg.fit(X, y)

X_to_predict = test.drop("status", axis=1)

Y_correct_prediction = test.status

y_predict = logreg.predict(X_to_predict)

accuracy = accuracy_score(Y_correct_prediction, y_predict)

balanced_accuracy = balanced_accuracy_score(Y_correct_prediction, y_predict)

recall = recall_score(Y_correct_prediction, y_predict)

f1_sc = f1_score(Y_correct_prediction, y_predict)

print("Accuracy: ", accuracy)
print("Balanced Accuracy: ", balanced_accuracy)
print("Recall: ", recall)
print("F1 Score: ", f1_sc)
