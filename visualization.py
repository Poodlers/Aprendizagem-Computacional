import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

conn = sqlite3.connect('bank_database.db')

# CLIENT :
# client_id
# birth_number
# gender
# district_id
client_df = pd.read_sql('SELECT * FROM client', conn)

# ACCOUNT : account_id
# district_id
# date
# frequency
account_df = pd.read_sql('SELECT * FROM account', conn)

# CARD_dev:
# card_id
# disp_id
# type
# issued
# card_dev_df = pd.read_sql('SELECT * FROM card_dev', conn)

# DISPOSITION:
# disp_id
# client_id
# account_id
# type
disp_df = pd.read_sql('SELECT * FROM disp', conn)
# DISTRICT:
# code
# name
# region
# no. of inhabitants
# no. of municipalities with inhabitants < 499
# no. of municipalities with inhabitants 500-1999
# no. of municipalities with inhabitants 2000-9999
# no. of municipalities with inhabitants >10000
# no. of cities
# ratio of urban inhabitants
# average salary
# unemploymant rate '95
# unemploymant rate '96
# no. of enterpreneurs per 1000 inhabitants
# no. of commited crimes '95
# no. of commited crimes '96
# district_df = pd.read_sql('SELECT * FROM district', conn)


# LOAN:
# loan_id
# account_id
# date
# amount
# duration
# payments
# status (VARIAVEL A PREVER)
loan_dev_df = pd.read_sql('SELECT * FROM loan_dev', conn)

# TRANSACTION:
# trans_id
# account_id
# date
# type
# operation
# amount
# balance
# k_symbol
# bank
# account
trans_dev_df = pd.read_sql('SELECT * FROM trans_dev', conn)

# Creating histogram
print(trans_dev_df.shape)

print(trans_dev_df.dropna(axis=0).shape)
