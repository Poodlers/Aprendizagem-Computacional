## excel_to_sql.py

- imported data from csv to python into dataframes
- exported data to sql
- added the separator format and header line to csv imports so that they are properly separated into columns
- created function make_into_date(var1) to tranform dates in the format YYMMDD into python datetime objects and after sql datetime objects
- converted the date collumns date in acc_df, loan_dev_df and trans_dev_df as well as date collumn issued from card_dev_df with the previous functions
- made it so in function make_into_date(var1) if the month is greater than 12 50 is subtracted from it
- converted the date column birth_number into a propper date in table client_df
- used list comprehension to properly assign genders in client_df based on if the month of the date of birth is greater than 12
- fixed bug where sometimes python assumed incorectly that the year was in the 2000s instead of 1900s

## analyser.py

- analyser.py is a tool used to extract information of columns within tables. It allows the user to switch between available tables and its columns and extract dat from them, like de average and variance of the elements of each column or the Pearson Correlation between two columns. If both columns are numeric, a scatter plot is generated, if only one is numeric and the other has less than 30 categories (this value can be changed) a bar plot is generated.
- To use simply type 'h' for a list of available commands.
- In the future, other statitistics may be added if there is a need for them.

## tests.py (Feature exatraction)

- Script that serves the purpose of calculating the average balance for every account taking into account all transactions

## visualization.py
