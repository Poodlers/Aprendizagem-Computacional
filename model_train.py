from numpy import NaN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import sqlite3

conn = sqlite3.connect('bank_database.db')

loan_dev_df = pd.read_sql('''SELECT * FROM account
     JOIN loan_dev ON loan_dev.account_id = account.account_id
    LEFT JOIN disp ON disp.account_id = account.account_id
    LEFT JOIN client ON client.client_id = disp.client_id
    LEFT JOIN district ON district.code = client.district_id
    WHERE disp.type = "OWNER"
''', conn)


loan_dev_df = loan_dev_df.replace('?', NaN)
loan_dev_df = loan_dev_df.dropna()

loan_dev_df = loan_dev_df.T.drop_duplicates().T

print(loan_dev_df)

train, test = train_test_split(loan_dev_df, test_size=0.2, random_state=0)

feature_cols = loan_dev_df.columns.drop(
    ["frequency", "date", "gender", "type", "name", "region", "birth_number"])

print(feature_cols)
X = train.loc[:, feature_cols]


y = train.status

# 2. instantiate model
logreg = LogisticRegression(max_iter=1000)

# 3. fit
logreg.fit(X, y)

X_to_predict = test.loc[:, feature_cols]

Y_correct_prediction = test.status

y_predict = logreg.predict(X_to_predict)

accuracy = accuracy_score(y_predict, Y_correct_prediction)

print(accuracy)
