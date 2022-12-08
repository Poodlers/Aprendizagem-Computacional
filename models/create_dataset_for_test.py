import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from encoder_one_hot import CategoricalOneHot


def process_dataset(database_path):
    # use ./test_database.db if testing for Kaggle
    # use '../bank_database.db' if training models
    conn = sqlite3.connect(database_path)

    loan_dev_df = pd.read_sql('''SELECT * FROM account
        JOIN loan_dev ON loan_dev.account_id = account.account_id
        LEFT JOIN disp ON disp.account_id = account.account_id
        LEFT JOIN client ON client.client_id = disp.client_id
        LEFT JOIN district ON district.code = client.district_id
        WHERE disp.type = "OWNER"
    ''', conn)

    # delete duplicated and NaN

    loan_dev_df = loan_dev_df.loc[:, ~loan_dev_df.columns.duplicated()]

    loan_dev_df.loc[loan_dev_df["no. of commited crimes '95 "] == '?',
                    "no. of commited crimes '95 "] = loan_dev_df["no. of commited crimes '96 "]

    loan_dev_df.loc[loan_dev_df["unemploymant rate '95 "] == '?',
                    "unemploymant rate '95 "] = loan_dev_df["unemploymant rate '96 "]

    # create age_at_loan - FEATURE ENGINEERING

    loan_dev_df = loan_dev_df.fillna(0)

    age_at_loan = []
    for i in range(loan_dev_df['birth_number'].size):
        try:
            loan_dev_df['date'][i]
        except:
            age_at_loan.append(np.NaN)
            continue

        age_at_loan.append((datetime.strptime(loan_dev_df['date'][i], '%Y-%m-%d').date(
        ) - datetime.strptime(loan_dev_df['birth_number'][i], '%Y-%m-%d').date()).days / 365.25)

    loan_dev_df = loan_dev_df.assign(age_at_loan=age_at_loan)

    # ENCODINGS

    categorical_data = ['frequency', 'gender', 'region']
    coh = CategoricalOneHot(list_key_words=categorical_data)
    loan_dev_df = coh.fit_transform(loan_dev_df)

    # FEATURE SELECTION

    excluded_cols = ["date", "type", "name", "birth_number", "district_id", "account_id", 'disp_id', 'client_id',
                     'code'
                     ]

    feature_cols = loan_dev_df.columns.drop(
        excluded_cols)

    loan_dev_df = loan_dev_df.drop(excluded_cols, axis=1)

    # print("Feature columns: ", feature_cols)
    return loan_dev_df, feature_cols


# ./test_data/test_database.db
# ../bank_database.db
process_dataset("./test_data/test_database.db")
