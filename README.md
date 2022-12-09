# Data Mining Project

## Team - G38

- Joana Mesquita - up201907878
- João Costa - up2019
- Miguel Freitas - up2019

We considered that we all worked equally in the elaboration of this project and therefore all deserve 1/3 when it come to the Individual factor.

# Files explanation

### csvs Folder

- Folder with the csvs given to elaborate the project

### [analyser.py](/analyser.py)

- script that displays some relevant feature's data such as mean, standard deviation, as well as correlation between variables

### [distributions_vars.py](/distributions_vars.py)

- script that displays feature's distributions

### [excel_to_sql.py](/excel_to_sql.py)

- script that converted the initial csv sheets into a more convenient sqlite database

### [pearson_correlation_coefficient.py](/pearson_correlation_coefficient.py)

- script that extracts pearson_correlation between features

### [data_understanding.ipynb](/data_understanding.ipynb)

- Python notebook containing various graphs made for data understanding

### [models/create_dataset_for_test.py](/models/create_dataset_for_test.py)

- script that prepares our data to be fed to our models, performing encoding and removal of useless features such as ids

### [models/encoder_one_hot.py](/models/encoder_one_hot.py)

- helper script that implemented OneHotEncoder to our specific needs, namely the creation of multiple features for each categorical feature

### [models/feature_selection.py](/models/feature_selection.py)

- script that applies both filter-based and wrapper-based feature selection methods

### [models/hyperparameter_tuning.py](/models/hyperparameter_tuning.py)

- script that helped us apply grid search with cross-validation in order to get the best parameters to our algorithms

### models/model_cross_validation.py

- script for model evaluation and metrics extraction

### models/model_train_test.py

- script for fitting algorithms and then predicting data

### models/test_data

- Data to predict and submit to kaggle
  - [models/test_data/pouplate_test_db.py](/models/test_data/pouplate_test_db.py) - Script to add the test data to a database ([models/test_data/test_database.db](/models/test_data/test_database.db))

### SMOTE_tests/smote.py

-
