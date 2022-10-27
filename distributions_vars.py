import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import math

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
    print("Active table: " + currentTableName + "      Active column : " +
          activeColumn)

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
activeTableColumnNames = getTableColumnNames(con, currentTableName)

activeColumn = activeTableColumnNames[0]

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
        activeTableColumnNames = getTableColumnNames(con, currentTableName)
        activeColumn = activeTableColumnNames[0]
        printCursorStatus()
    # command 1 is to change current column 1
    elif command == '1':
        currentColumnIndex = activeTableColumnNames.index(activeColumn)
        if currentColumnIndex == len(activeTableColumnNames) - 1:
            activeColumn = activeTableColumnNames[0]
        else:
            activeColumn = activeTableColumnNames[currentColumnIndex+1]
        printCursorStatus()
    # command q is to quit
    elif command == 'q' or command == 'quit' or command == 'exit':
        quit = True
    # command e is to extract results and graphs from current table and columns
    elif command == 'a' or command == 'analyse':
        res = con.execute("SELECT \"" + activeColumn +
                          "\" FROM " + currentTableName + ";")
        column1Data = []
        for result in res.fetchall():
            column1Data.append(result[0])
        colum1IsNumeric = checkArrayValuesFloat(column1Data)
        if colum1IsNumeric:
            printArrayStatistics(column1Data, "Column '" + activeColumn + "'")
        else:
            print("Column 1 is not numeric\n")
        if colum1IsNumeric:
            print("Showing graph\n")
            fig, (ax) = plt.subplots()
            ax.set_xlabel(activeColumn)
            ax.set_ylabel("frequency")
            plt.title(activeColumn + " distribution")
            plt.hist(column1Data, bins=20)
            plt.axhline(0)
            plt.show()
        printCursorStatus()
    elif command == 'help' or command == 'h':
        print("-------------------- Available Commands --------------------\n")
        print(" t - Change active table")
        print(" 1 - Change active first column")
        print(" a - Print variable distribution ")
        print(" q - Quit")
        print(" h - Help Screen")
        print("\n------------------------------------------------------------")
    else:
        print("Unkown command, type help for list of commands")
