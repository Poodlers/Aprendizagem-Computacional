import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import math

import pandas as pd

MAX_CATEGORIES = 15

# Retrieves the name of the columns in the database


def getTableColumnNames(connection, tableName):
    columnNames = []
    # Get information from a table
    res = connection.execute("PRAGMA table_info(" + tableName + ");")
    for result in res.fetchall():  # Create a list with all column names
        columnNames.append(result[1])
    return columnNames

# Prints the current table, column 1 and column 2


def printCursorStatus():
    print("Active table: " + currentTableName + "      Active column 1: " +
          activeColumn1 + "      Active column 2: " + activeColumn2)

# Prints the statistics of an array a. Title is for displaying purposes only


def printArrayStatistics(a, title):
    print("----- Statistics for " + title + " -----")
    max = np.max(a)
    min = np.min(a)
    average = np.average(a)
    variance = np.var(a)
    std = np.std(a)
    print("Max: " + str(max) + "   Min: " + str(min))
    print("Average: " + str(average))
    print("Variance: " + str(variance))
    print("Standard Deviation: " + str(std))
    print("")


def printTwoArrayStatistics(a, b, title1, title2):
    average1 = np.average(a)
    average2 = np.average(b)
    std1 = np.std(a)
    std2 = np.std(b)
    if len(a) == len(b):
        print("----- Statistics between " +
              title1 + " and " + title2 + " -----")
        sum = 0
        for i in range(len(a)):
            sum += (a[i] - average1) * (b[i] - average2)
        covariance = sum/(len(a)-1)
        correlation = covariance/(std1*std2)
        print("Covariance: " + str(covariance))
        print("Correlation: " + str(correlation))
        if abs(correlation) > 0.3:
            print("    <-----")
        else:
            print("")
    print("")

# Checks if the contents of an array are all numeric


def checkArrayValuesFloat(arr):
    for element in arr:
        try:
            float(element)
        except ValueError:
            return False
    return True

# Check how many categories a column has


def checkCategoryNumber(column):
    categories = []
    for element in column:
        try:
            categories.index(element)
        except ValueError:
            categories.append(element)
    return len(categories)


con = sqlite3.connect("bank_database.db")

c = con.cursor()

loan_dev_df = pd.read_sql('''SELECT * FROM account
     JOIN loan_dev ON loan_dev.account_id = account.account_id
    LEFT JOIN disp ON disp.account_id = account.account_id
    LEFT JOIN client ON client.client_id = disp.client_id
    LEFT JOIN district ON district.code = client.district_id
    WHERE disp.type = "OWNER"
''', con)

activeTableColumnNames = loan_dev_df.columns

# Get table names
res = con.execute("SELECT name FROM sqlite_master WHERE type='table';")
tableNames = []
for result in res.fetchall():
    tableNames.append(result[0])
# Set current active table
if len(tableNames) == 0:
    print("No tables available!")
    exit()
currentTableName = tableNames[0]

# Get table's column names

activeColumn1 = activeTableColumnNames[0]
activeColumn2 = activeTableColumnNames[1]

# print current table and columns
printCursorStatus()

quit = False
# start commands loop
while not quit:
    command = input()
    # command t is to change active table
    if command == 't':
        currentTableIndex = tableNames.index(currentTableName)
        if currentTableIndex == len(tableNames) - 1:
            currentTableName = tableNames[0]
        else:
            currentTableName = tableNames[currentTableIndex+1]

        activeColumn1 = activeTableColumnNames[0]
        activeColumn2 = activeTableColumnNames[1]
        printCursorStatus()
    # command 1 is to change current column 1
    elif command == '1':
        currentColumnIndex = activeTableColumnNames.index(activeColumn1)
        if currentColumnIndex == len(activeTableColumnNames) - 1:
            activeColumn1 = activeTableColumnNames[0]
        else:
            activeColumn1 = activeTableColumnNames[currentColumnIndex+1]
        printCursorStatus()
    # command 2 is to change current column 2
    elif command == '2':
        currentColumnIndex = activeTableColumnNames.index(activeColumn2)
        if currentColumnIndex == len(activeTableColumnNames) - 1:
            activeColumn2 = activeTableColumnNames[0]
        else:
            activeColumn2 = activeTableColumnNames[currentColumnIndex+1]
        printCursorStatus()
    # command q is to quit
    elif command == 'q' or command == 'quit' or command == 'exit':
        quit = True
    # command e is to extract results and graphs from current table and columns
    elif command == 'a' or command == 'analyse':
        res = con.execute(f'''SELECT {activeColumn1}, {activeColumn2} FROM account
                JOIN loan_dev ON loan_dev.account_id = account.account_id
                LEFT JOIN disp ON disp.account_id = account.account_id
                LEFT JOIN client ON client.client_id = disp.client_id
                LEFT JOIN district ON district.code = client.district_id
                WHERE disp.type = "OWNER"
        ''', con)
        column1Data = []
        column2Data = []
        for result in res.fetchall():
            column1Data.append(result[0])
            column2Data.append(result[1])
        colum1IsNumeric = checkArrayValuesFloat(column1Data)
        colum2IsNumeric = checkArrayValuesFloat(column2Data)
        if colum1IsNumeric:
            printArrayStatistics(column1Data, "Column '" + activeColumn1 + "'")
        else:
            print("Column 1 is not numeric\n")
        if colum2IsNumeric:
            printArrayStatistics(column2Data, "Column '" + activeColumn2 + "'")
        else:
            print("Column 2 is not numeric\n")
        if colum1IsNumeric and colum2IsNumeric:
            printTwoArrayStatistics(
                column1Data, column2Data, "Column 1", "Column 2")
            print("Showing graphs\n")
            fig, (ax) = plt.subplots()
            ax.set_xlabel(activeColumn1)
            ax.set_ylabel(activeColumn2)
            plt.title(activeColumn1 + " vs " + activeColumn2)
            plt.scatter(column1Data, column2Data)
            plt.axhline(0)
            plt.show()
        elif colum2IsNumeric:
            if checkCategoryNumber(column1Data) < MAX_CATEGORIES:
                fig, (ax) = plt.subplots()
                ax.set_xlabel(activeColumn1)
                ax.set_ylabel(activeColumn2)
                plt.title(activeColumn1 + " vs " + activeColumn2)
                plt.bar(column1Data, column2Data)
                plt.axhline(0)
                plt.show()
        elif colum1IsNumeric:
            if checkCategoryNumber(column2Data) < MAX_CATEGORIES:
                fig, (ax) = plt.subplots()
                ax.set_xlabel(activeColumn1)
                ax.set_ylabel(activeColumn2)
                plt.title(activeColumn1 + " vs " + activeColumn2)
                plt.barh(column1Data, column2Data)
                plt.axhline(0)
                plt.show()
        printCursorStatus()
    elif command == 'help' or command == 'h':
        print("-------------------- Available Commands --------------------\n")
        print(" t - Change active table")
        print(" 1 - Change active first column")
        print(" 2 - Change active second column")
        print(" a - Analyse relationship between columns and extract data")
        print(" q - Quit")
        print(" h - Help Screen")
        print("\n------------------------------------------------------------")
    else:
        print("Unkown command, type help for list of commands")
