# Business Understanding

### BU: analysis of requirements with the end user

Our model should be able to guess correctly at least 90% of the outcomes of loans to be given.

### BU: definition of business goals

The goal is to develop a model to aid the end user in deciding wether a specific client is going to pay back a loan he is applying to and, therefore, if this loan is worth lending. The model should take into account a set of characteristics of the client and use them to make a decision.

### BU: translation of business goals into data mining goals

A data mining model needs to be created to process the data. This model will be trained with data given by the bank that contanis information about past lent loans, account informations, geographical statistics, among others. By training the model with this data we should be able to develop the behavior we aim to achieve. In order to ensure that our model is indeed guessing the loan results right, we need to be careful not to use the whole data for training but to save some of it for testing purposes. Since that an unpaid loan is significantly worse for the bank than an unconceeded loan, our model should be trained with processes that pay close attention to _False Positives_ in deteriment of \_False Negatives, and it should be assessed with test using the same philosophy.

# Data Understanding

### DU: diversity of statistical methods

We analysed multiple fields of different tables, calculating their average, variance, standard deviation and looking for outliers, through analytical and graphical methods. We also made an analysis between diferent columns, caclulating their Pearson correlation.

### DU: complexity of statistical methods

The biggest complexity that we implemented was creating a program that allowed us to quickly analyse the data (analyser.py) and extract all its statistical information.

### DU: interpretation of results of statistical methods

It was very dificult to find clear evidence of strong correlation between different fields, but we were able to find, for example, that the bigger the loan is, the less it is prone to fail.

### DU: knowledge extraction from results of statistical methods

### DU: diversity of plots

We created mostly scatter plots and bar plots.

### DU: complexity of plots

The complexity of the plots is relatively low, but that didn't make us less aware of the data before us, it rather made it easier for us to understand the nature of the data

### DU: presentation

### DU: interpretation of plots

# Data Processing

### DP: data integration

### DP: assessment of dimensions of data quality

2 dimensions (usually accuracy and completeness)

3 dimensions (... consistency or uniqueness)

4 dimensions (... consistency and uniqueness)

### DP (cleaning): redundancy

removal of some redundant attributes: training the module without the following

["date", "type", "name", "region", "birth_number", "no. of municipalities with inhabitants < 499 ",
"no. of municipalities with inhabitants 500-1999", "no. of municipalities with inhabitants 2000-9999 ",
"no. of municipalities with inhabitants >10000 ", "loan_id", "district_id", "account_id", "status"
]

### DP (cleaning): missing data

Currently dropping the missing values (only model that exists is Logistic Regression which doesnt accept missing values anyway)

`train = train.dropna() `

### DP (cleaning): outliers

### DP: data transformation for compatibility with algorithms

discretization of gender and frequency variables to feed our LogisticRegression

`age_dict = {"M": 0, "F": 1} frequency_dict = {"monthly issuance": 0, "issuance after transaction": 1, "weekly issuance": 2}`

### DP: feature engineering from tabular data

We attempted to extract the average balance when making transactions and average transaction amount to each account in order to maybe find correlation between the data.

### DP: sampling for domain-specific purposes

### DP: sampling for development

### DP: imbalanced data

### DP: feature selection

# Algorithms - training a model

### descriptive: diversity of algorithms

Tested LogisticRegression algorithm

### descriptive: parameter tuning
