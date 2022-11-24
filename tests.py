import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import math


con = sqlite3.connect("bank_database.db")

c = con.cursor()

# Get table names

'''
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

'''



'''
res1 = con.execute("SELECT account_id FROM account")
res1 = res1.fetchall()
i=0
nulls = 0
not_nulls = 0
average_balance = []
for account in res1:
    res2 = con.execute("SELECT amount FROM trans_dev WHERE account_id == " + str(account[0]))
    res2 = res2.fetchall()
    if len(res2) == 0:
        res = con.execute("UPDATE account SET min_transaction_amount = NULL, max_transaction_amount = NULL WHERE account_id == " + str(account[0]))
        nulls += 1
    else:
        not_nulls +=1
        min = 99999999999999
        max = 0
        print(res2)
        for balance in res2:
            if balance[0] > max:
                max = balance[0]
            if balance[0] < min:
                min = balance[0]
        res = con.execute("UPDATE account SET min_transaction_amount = " + str(min) + ", max_transaction_amount = " + str(max) + " WHERE account_id == " + str(account[0]))
    i+=1
    print("{:.2f}".format(i/45) + "% (" + str(i) + "/4500)  " + str(not_nulls) + " not nulls and " + str(nulls) + " nulls",end="\r")
con.commit()       

'''

res1 = con.execute("SELECT average_balance FROM account WHERE average_balance IS NOT NULL")
res1 = res1.fetchall()
print(len(res1))
total = 0
count = 0
for balance in res1:
    total += balance[0]
    count += 1
avg = total/count


con.execute("UPDATE account SET average_balance = " + str(avg) + " WHERE average_balance IS NULL")

con.commit()

    
