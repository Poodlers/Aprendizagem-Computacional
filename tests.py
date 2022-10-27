import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import math



con = sqlite3.connect("bank_database.db")

c = con.cursor()

# Get table names
res = con.execute("SELECT account_id,amount FROM trans_dev")
amount = []
account_id = []
for result in res.fetchall():
    amount.append(result[1])
    account_id.append(result[0])
trans_num = {}
averageTransAmount = {}
for index,account in enumerate(account_id):
    record = averageTransAmount.get(account,-1)
    if record == -1:
        averageTransAmount[account] = amount[index]
        trans_num[account] = 1
    else:
        averageTransAmount[account] += amount[index]
        trans_num[account] += 1
for account in averageTransAmount:
    averageTransAmount[account] = averageTransAmount[account] / trans_num[account]

for account in averageTransAmount:
    res = con.execute("UPDATE account SET average_trans_amount = " + str(averageTransAmount[account]) + " WHERE account_id = " + str(account))
    con.commit()

