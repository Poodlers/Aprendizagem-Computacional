import numpy as np
import matplotlib.pyplot as plt

import sqlite3

con = sqlite3.connect("bank_database.db")

c = con.cursor()

res = c.execute("SELECT amount, payments, duration FROM loan_dev;")

data = res.fetchall()

payments = []
amount = []
duration = []

for result in data:
    amount.append(result[0])
    payments.append(result[1])
    duration.append(result[2])



a = np.array([amount,payments,duration])


my_rho = np.corrcoef(a)

print(my_rho)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

ax[0].scatter(amount,duration)
ax[0].title.set_text('Correlation = ' + "{:.2f}".format(my_rho[0,2]))
ax[0].set(xlabel='amount',ylabel='duration')

ax[1].scatter(amount,payments)
ax[1].title.set_text('Correlation = ' + "{:.2f}".format(my_rho[0,1]))
ax[1].set(xlabel='amount',ylabel='payments')

ax[2].scatter(payments,duration)
ax[2].title.set_text('Correlation = ' + "{:.2f}".format(my_rho[1,2]))
ax[2].set(xlabel='payments',ylabel='duration')

fig.subplots_adjust(wspace=.4)    
plt.show()