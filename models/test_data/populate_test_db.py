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


acc_df = pd.read_csv('../../csvs/account.csv', sep=";", header=0)
card_df = pd.read_csv('./card_comp.csv', sep=";", header=0)
client_df = pd.read_csv('../../csvs/client.csv', sep=";", header=0)
disp_df = pd.read_csv('../../csvs/disp.csv', sep=";", header=0)
district_df = pd.read_csv('../../csvs/district.csv', sep=";", header=0)
loan_df = pd.read_csv('./loan_comp.csv', sep=";", header=0)
trans_df = pd.read_csv('./trans_comp.csv', sep=";", header=0)

acc_df['date'] = acc_df['date'].apply(make_into_date)
loan_df['date'] = loan_df['date'].apply(make_into_date)


trans_df['date'] = trans_df['date'].apply(make_into_date)
card_df['issued'] = card_df['issued'].apply(make_into_date)
gender = ["M" if int(str(client_df['birth_number'][i])[2:4]) <=
          12 else "F" for i in range(client_df['birth_number'].size)]
new_client_df = client_df.assign(gender=gender)


new_client_df['birth_number'] = new_client_df['birth_number'].apply(
    make_into_date)

conn = sqlite3.connect('test_database.db')
c = conn.cursor()

acc_df.to_sql('account', con=conn, if_exists='replace',
              index=False, index_label='account_id')
card_df.to_sql('card_dev', con=conn, if_exists='replace',
               index=False, index_label='card_id')
new_client_df.to_sql('client', con=conn, if_exists='replace',
                     index=False, index_label='client_id')
disp_df.to_sql('disp', con=conn, if_exists='replace',
               index=False, index_label='disp_id')
district_df.to_sql('district', con=conn, if_exists='replace',
                   index=False, index_label='code')
loan_df.to_sql('loan_dev', con=conn, if_exists='replace',
               index=False, index_label='loan_id')
trans_df.to_sql('trans_dev', con=conn, if_exists='replace',
                index=False, index_label='trans_id')


conn.commit()

print(c.execute("SELECT account_id FROM account;"))


conn.close()
