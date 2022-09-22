from sqlalchemy import create_engine
import pandas as pd
import sqlite3

acc_df = pd.read_csv('csvs/account.csv')
card_dev_df = pd.read_csv('csvs/card_dev.csv')
client_df = pd.read_csv('csvs/client.csv')
disp_df = pd.read_csv('csvs/disp.csv')
district_df = pd.read_csv('csvs/district.csv')
loan_dev_df = pd.read_csv('csvs/loan_dev.csv')
trans_dev_df = pd.read_csv('csvs/trans_dev.csv')

conn = sqlite3.connect('bank_database.db')
c = conn.cursor()

acc_df.to_sql('account', con=conn, if_exists='replace', index=False, index_label='account_id')
card_dev_df.to_sql('card_dev', con=conn, if_exists='replace', index=False, index_label='card_id')
client_df.to_sql('client', con=conn, if_exists='replace', index=False, index_label='client_id')
disp_df.to_sql('disp', con=conn, if_exists='replace', index=False, index_label='disp_id')
district_df.to_sql('district', con=conn, if_exists='replace', index=False, index_label='code')
loan_dev_df.to_sql('loan_dev', con=conn, if_exists='replace', index=False, index_label='loan_id')
trans_dev_df.to_sql('trans_dev', con=conn, if_exists='replace', index=False, index_label='trans_id')


conn.commit()


conn.close()
