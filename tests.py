import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import math


con = sqlite3.connect("./models/test_data/test_database.db")

c = con.cursor()

# Get table names

con.execute("ALTER TABLE account ADD COLUMN average_trans_amount")
con.execute("ALTER TABLE account ADD COLUMN average_balance")
con.execute("ALTER TABLE account ADD COLUMN min_balance")
con.execute("ALTER TABLE account ADD COLUMN max_balance")
con.execute("ALTER TABLE account ADD COLUMN min_transaction_amount")
con.execute("ALTER TABLE account ADD COLUMN max_transaction_amount")


res = con.execute("SELECT account_id,amount FROM trans_dev")
amount = []
account_id = []
for result in res.fetchall():
    amount.append(result[1])
    account_id.append(result[0])
trans_num = {}
averageTransAmount = {}
for index, account in enumerate(account_id):
    record = averageTransAmount.get(account, -1)
    if record == -1:
        averageTransAmount[account] = amount[index]
        trans_num[account] = 1
    else:
        averageTransAmount[account] += amount[index]
        trans_num[account] += 1
for account in averageTransAmount:
    averageTransAmount[account] = averageTransAmount[account] / \
        trans_num[account]

for account in averageTransAmount:
    res = con.execute("UPDATE account SET average_trans_amount = " +
                      str(averageTransAmount[account]) + " WHERE account_id = " + str(account))
    con.commit()


res1 = con.execute("SELECT account_id FROM account")
res1 = res1.fetchall()
i = 0
nulls = 0
not_nulls = 0
average_balance = []
for account in res1:
    res2 = con.execute(
        "SELECT amount,balance FROM trans_dev WHERE account_id == " + str(account[0]))
    res2 = res2.fetchall()
    if len(res2) == 0:
        res = con.execute(
            "UPDATE account SET min_transaction_amount = NULL, max_transaction_amount = NULL WHERE account_id == " + str(account[0]))
        nulls += 1
    else:
        not_nulls += 1
        min_am = 99999999999999
        max_am = 0
        min_bal = 9999999999999
        max_bal = -999999999999
        total_am = 0
        total_bal = 0
        trans_count = 0
        for trans in res2:  # trans[0] -> amount  trans[1] -> balance
            if trans[0] > max_am:
                max_am = trans[0]
            if trans[0] < min_am:
                min_am = trans[0]
            if trans[1] > max_bal:
                max_bal = trans[1]
            if trans[1] < min_bal:
                min_bal = trans[1]
            total_am += trans[0]
            total_bal += trans[1]
            trans_count += 1
        avg_am = total_am/trans_count
        avg_bal = total_bal/trans_count
        res = con.execute("UPDATE account SET min_transaction_amount = " + str(min_am) +
                          ", max_transaction_amount = " + str(max_am) +
                          ", min_balance = " + str(min_bal) +
                          ", max_balance = " + str(max_bal) +
                          ", average_trans_amount = " + str(avg_am) +
                          ", average_balance = " + str(avg_bal) +
                          " WHERE account_id == " + str(account[0]))
    i += 1
    print("{:.2f}".format(i/45) + "% (" + str(i) + "/4500)  " +
          str(not_nulls) + " not nulls and " + str(nulls) + " nulls", end="\r")
con.commit()

res1 = con.execute(
    "SELECT average_balance FROM account WHERE average_balance IS NOT NULL")
res1 = res1.fetchall()
total = 0
count = 0
for balance in res1:
    total += balance[0]
    count += 1
avg = total/count
con.execute("UPDATE account SET average_balance = " +
            str(avg) + " WHERE average_balance IS NULL")

con.commit()
