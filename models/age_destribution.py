import pandas as pd
from datetime import datetime
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

conn = sqlite3.connect('./bank_database.db')

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

age_at_loan = []
for i in range(loan_dev_df['birth_number'].size):
    try:
        loan_dev_df['date'][i]
    except:
        age_at_loan.append(np.NaN)
        continue

    age_at_loan.append((datetime.strptime(loan_dev_df['date'][i], '%Y-%m-%d').date(
    ) - datetime.strptime(loan_dev_df['birth_number'][i], '%Y-%m-%d').date()).days / 365.25)


plt.hist(age_at_loan, bins=50)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
plt.show()
