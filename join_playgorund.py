from numpy import sort
import pandas as pd
import sqlite3

conn = sqlite3.connect('bank_database.db')
'''
LEFT JOIN disp ON disp.account_id = account.account_id
    LEFT JOIN client ON client.client_id = disp.client_id
    LEFT JOIN district ON district.code = client.district_id
'''

loan_dev_account_df = pd.read_sql('''SELECT account.account_id, loan_dev.loan_id FROM loan_dev
     JOIN account ON loan_dev.account_id = account.account_id
    
''', conn)

loan_df = pd.read_sql('''SELECT * FROM loan_dev
''', conn)
