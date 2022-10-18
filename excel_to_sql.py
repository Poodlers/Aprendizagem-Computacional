
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd
import sqlite3


def make_into_date(var1):
    var = str(var1)

    if int(var[2:4]) > 12:
        var = var[0:2] + str(int(var[2:4]) - 50).zfill(2) + var[4:6]
    time = var[4:6] + "-" + var[2:4] + "-19" + var[0:2]
    return datetime.strptime(time, '%d-%m-%Y').date()


acc_df = pd.read_csv('csvs/account.csv', sep=";", header=0)
card_dev_df = pd.read_csv('csvs/card_dev.csv', sep=";", header=0)
client_df = pd.read_csv('csvs/client.csv', sep=";", header=0)
disp_df = pd.read_csv('csvs/disp.csv', sep=";", header=0)
district_df = pd.read_csv('csvs/district.csv', sep=";", header=0)
loan_dev_df = pd.read_csv('csvs/loan_dev.csv', sep=";", header=0)
trans_dev_df = pd.read_csv('csvs/trans_dev.csv', sep=";", header=0)

acc_df['date'] = acc_df['date'].apply(make_into_date)
loan_dev_df['date'] = loan_dev_df['date'].apply(make_into_date)


trans_dev_df['date'] = trans_dev_df['date'].apply(make_into_date)
card_dev_df['issued'] = card_dev_df['issued'].apply(make_into_date)
gender = ["M" if int(str(client_df['birth_number'][i])[2:4]) <=
          12 else "F" for i in range(client_df['birth_number'].size)]
new_client_df = client_df.assign(gender=gender)


new_client_df['birth_number'] = new_client_df['birth_number'].apply(
    make_into_date)

conn = sqlite3.connect('bank_database.db')
c = conn.cursor()

acc_df.to_sql('account', con=conn, if_exists='replace',
              index=False, index_label='account_id')
card_dev_df.to_sql('card_dev', con=conn, if_exists='replace',
                   index=False, index_label='card_id')
new_client_df.to_sql('client', con=conn, if_exists='replace',
                     index=False, index_label='client_id')
disp_df.to_sql('disp', con=conn, if_exists='replace',
               index=False, index_label='disp_id')
district_df.to_sql('district', con=conn, if_exists='replace',
                   index=False, index_label='code')
loan_dev_df.to_sql('loan_dev', con=conn, if_exists='replace',
                   index=False, index_label='loan_id')
trans_dev_df.to_sql('trans_dev', con=conn, if_exists='replace',
                    index=False, index_label='trans_id')


conn.commit()

print(c.execute("SELECT account_id FROM account;"))


conn.close()
