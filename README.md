# Business Understanding

### BU: analysis of requirements with the end user

### BU: definition of business goals

    The goal is to develop a model to aid the end user in deciding wether a specific client is going to pay back a loan he is applying to and, therefore, if this loan is worth lending. The model should take into account a set of characteristics of the client and use them to make a decision.

### BU: translation of business goals into data mining goals

    A data mining model needs to be created to process the data. This mofel will be trained with data given by the bank that contanis information about past lent loans, account informations, geographical statistics, among others. By training the model with this data we should be able to develop the behavior we aim to achieve. In order to ensure that our model is indeed guessing the loan results right, we need to be careful not to use the whole data for training but to save some of it for testing purposes. Since that an un

# Data Understanding

### DU: diversity of statistical methods

### DU: complexity of statistical methods

### DU: interpretation of results of statistical methods

### DU: knowledge extraction from results of statistical methods

### DU: diversity of plots

### DU: complexity of plots

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

### DP: sampling for domain-specific purposes

### DP: sampling for development

### DP: imbalanced data

### DP: feature selection

# Algorithms - training a model

### descriptive: diversity of algorithms

Tested LogisticRegression algorithm

### descriptive: parameter tuning
