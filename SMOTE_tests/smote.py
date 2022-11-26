import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import math
import random
from random import randint
from math import sqrt
from support_functions import *

#Give it a date in the format "yyyy-mm-dd" and it returns [yyyy,mm,dd]
def parseDate(date):
    tokens = date.split("-")
    tokens2 = []
    for i in range(len(tokens)):
        tokens2.append(int(tokens[i]))
    return tokens2

#values -> [12, 378.93, 902]; vector -> [[14,432.3,567],[16,763.3,876],...]; n -> how many to get
def getNearestNeighbours(values,vector, n):
    answer = []
    if len(vector) <= n:
        for i in range(len(vector)):
            answer.append(i)
        return answer
    #For each neighbour to find
    for i in range(n):
        closestIndex = 0
        closestDifference = 9999999
        #Iterate through all rows
        for j in range(len(vector)):
            if j in answer:
                continue
            #Find the distance from that row to values row
            distance = 0
            for k in range(len(values)):
                distance += pow(values[k] - vector[j][k],2)
            distance = sqrt(distance)  
            if distance < closestDifference:
                closestDifference = distance
                closestIndex = j
        answer.append(closestIndex)
    return answer


con = sqlite3.connect("bank_database.db")
c = con.cursor()

#Get population of district to generate random district according to their population
res1 = con.execute("""SELECT "no. of inhabitants"  FROM district""")
res1 = res1.fetchall()
population = []
for district in res1:
    population.append(district[0])


def getRandomDistrict():
    randomPerson = randint(0,10309137)
    district = 0
    while population[district] < randomPerson:
        randomPerson -= population[district]
        district += 1
    return district+1



#APPLYING SMOTE
loans_to_generate = 50 #There are 282 successful loans and 46 failed loans

#Getting all amounts and account ids of failed loand
failed_loans = con.execute("SELECT amount, account_id FROM loan_dev WHERE status == -1").fetchall()
#Normalizing these loans
normalized_failed_loans = []
for loan in failed_loans:
    normalized_failed_loans.append([getNormalizedLoanAmount(loan[0]),loan[1]])
#Getting all accounts and normalizing them
accounts = con.execute("SELECT * FROM account")
normalized_accounts = []
for account in accounts:
    if account[0] == None or account[1] == None or account[2] == None or account[3] == None or account[4] == None or account[5] == None or account[6] == None or account[7] == None or account[8] == None or account[9] == None:
        continue
    normalized_accounts.append([account[0],account[1],account[2],account[3],
                getNormalizedAverageTransAmount(account[4]),getNormalizedAverageBalance(account[5]),
                getNormalizedMinBalance(account[6]),getNormalizedMaxBalance(account[7]),
                getNormalizedMinTransAmount(account[8]),getNormalizedMaxTransAmount(account[9])])
#Function that returns a normalized id
def findNormalizedAccount(account_id):
    for account in normalized_accounts:
        if account[0] == account_id:
            return account
    return None
#Generated a table with normalized values: loan-amount, avg_trans_am, avg_bal, min_bal, max_bal, min_trans_am, max_trans_am
normalizedValues = []
for i in range(len(failed_loans)):
    normAccount = findNormalizedAccount(normalized_failed_loans[i][1])
    normalizedValues.append([normalized_failed_loans[i][0],normAccount[4],normAccount[5],normAccount[6],normAccount[7],normAccount[8],normAccount[9]])


k = 5
delta = 0.2

for i in range(loans_to_generate):
    basis_loan = random.choice(normalized_failed_loans)
    basis_acc = findNormalizedAccount(int(basis_loan[1]))
    if basis_acc == None:
        print("FAILED TO FIND ACCOUNT")
    values = [basis_loan[0],basis_acc[4],basis_acc[5],basis_acc[6],basis_acc[7],basis_acc[8],basis_acc[9]]
    target_loan = getNearestNeighbours(values,normalizedValues,k)[k-1]
    newValues = []
    for j in range(len(values)):
        newValues.append(values[j] + (normalizedValues[target_loan][j] - values[j]) * delta)
    new_loan_id = 8000 + i
    new_account_id = 4501 + i
    new_loan_month = randint(0,12)
    new_loan_date = str(getRandomYear()) +"-" + "{:02d}".format(new_loan_month)+ "-" + "{:02d}".format(getRandomDay(new_loan_month))
    new_loan_duration = getRandomDuration()
    new_loan_status = -1
    new_loan_payments = getDenormalizedLoanAmount(newValues[0])/new_loan_duration
    new_account_month = randint(1,11)
    new_account_date = str(getRandomYear()) +"-" + "{:02d}".format(new_account_month)+ "-" + "{:02d}".format(getRandomDay(new_account_month))
    q = "INSERT INTO loan_dev (loan_id, account_id, date, amount, duration, payments, status) VALUES (" + str(new_loan_id) +", " + str(new_account_id) + ", '" + new_loan_date + "', " + str(getDenormalizedLoanAmount(newValues[0])) + ", " + str(new_loan_duration) + ", " + str(new_loan_payments) +", " + str(new_loan_status) + ")"
    print(q)
    con.execute(q)
    q = "INSERT INTO account (account_id, district_id, frequency, date, average_trans_amount, average_balance, min_balance, max_balance, min_transaction_amount, max_transaction_amount) VALUES (" + str(new_account_id) + ", " + str(getRandomDistrict()) + ", '" + getRandomFrequency() + "', '" + new_account_date + "', " + str(getDenormalizedAverageTransAmount(newValues[1])) + ", " + str(getDenormalizedAverageBalance(newValues[2])) + ", " + str(getDenormalizedMinBalance(newValues[3])) + ", " + str(getDenormalizedMaxBalance(newValues[4])) + ", " + str(getDenormalizedMinTransAmount(newValues[5])) + ", " + str(getDenormalizedMaxTransAmount(newValues[6])) + ")"
    print(q)
con.commit()


#con.execute("""INSERT INTO loan_dev (duration, payments, age_at_loan, amount, account_id) """)