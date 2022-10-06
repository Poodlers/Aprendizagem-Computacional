import sqlite3
from matplotlib import pyplot as plt
import numpy as np

con = sqlite3.connect("bank_database.db")

c = con.cursor()

completed_loans = np.zeros(100)
failed_loans = np.zeros(100)
total_loans = np.zeros(100)
x = np.arange(100)

res = c.execute("SELECT amount,status FROM loan_dev;")
for result in res.fetchall():
    amount = result[0]
    status = result[1]
    print("Got amount " + str(amount)+ ", putting on index " + str(int(int(amount)/1000)))
    if status == 1:
        completed_loans[int(int(amount)/10000)] +=1
        total_loans[int(int(amount)/10000)] +=1
    else:
        failed_loans[int(int(amount)/10000)] -=1
        total_loans[int(int(amount)/10000)] -=1
for i in range(100):
    if completed_loans[i] - failed_loans[i] > 0:
        total_loans[i] = total_loans[i] / (completed_loans[i] - failed_loans[i])
print("COMPLETED LOANS")
print(completed_loans)
print("FAILED LOANS")
print(failed_loans)
print("TOTAL LOANS")
print(total_loans)
plt.figure()
fig , (ax) = plt.subplots()
plt.xlim(0,60)
ax.set_ylabel("Number of completed/failed loans")
ax.set_xlabel("Loan's amount")
plt.bar(x,completed_loans, color='green')
plt.bar(x,failed_loans, color='red')
plt.title("Completed and failed loans by amount")
plt.axhline(0)
plt.show()
fig , (ax) = plt.subplots()
plt.xlim(0,60)
ax.set_ylabel("Normalized difference between completed and failed loans")
ax.set_xlabel("Loan's amount")
plt.bar(x,total_loans, color='yellow')
plt.title("Completed and failed loans by amount")
plt.axhline(0)
plt.show()