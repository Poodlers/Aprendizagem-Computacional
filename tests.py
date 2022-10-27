import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import math



con = sqlite3.connect("bank_database.db")

c = con.cursor()

# Get table names
res = con.execute("SELECT account_id,balance FROM trans_dev")
balance = []
account_id = []
for result in res.fetchall():
    balance.append(result[1])
    account_id.append(result[0])
trans_num = {}
averageBalance = {}
for index,account in enumerate(account_id):
    record = averageBalance.get(account,-1)
    if record == -1:
        averageBalance[account] = balance[index]
        trans_num[account] = 1
    else:
        averageBalance[account] += balance[index]
        trans_num[account] += 1
for account in averageBalance:
    averageBalance[account] = averageBalance[account] / trans_num[account]
print(averageBalance)