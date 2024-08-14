import gpboost as gpb
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.inspection import permutation_importance
import numpy as np


def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")


# Read the characteristics of the training and test sets
Year_train = pd.read_csv('./data/order_split/Year_train.csv').drop('ID', axis=1)
Year_test = pd.read_csv('./data/order_split/Year_test.csv').drop('ID', axis=1)
X_train = pd.read_csv('./data/order_split/X_train_df.csv').drop('ID', axis=1)
X_test = pd.read_csv('./data/order_split/X_test_df.csv').drop('ID', axis=1)
X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
# Read the training set and test set labels
y_train = pd.read_csv('./data/order_split/y_train_df.csv').drop('ID', axis=1)
y_test = pd.read_csv('./data/order_split/y_test_df.csv').drop('ID', axis=1)
y = pd.concat([y_train, y_test], axis=0, ignore_index=True)
# Read the group information of the training set and the test set
CID_train = pd.read_csv('./data/order_split/CID_train.csv').drop('ID', axis=1)
CID_test = pd.read_csv('./data/order_split/CID_test.csv').drop('ID', axis=1)

Year = pd.concat([Year_train, Year_test], axis=0, ignore_index=True)
CID = pd.concat([CID_train, CID_test], axis=0, ignore_index=True)

group_train = pd.concat([Year_train, CID_train], axis=1)
group_test = pd.concat([Year_test, CID_test], axis=1)
group = pd.concat([Year, CID], axis=1)
# Create a GPModel
gp_model = gpb.GPModel(group_data=group_train)

# Define parameters
# params = {'learning_rate': 0.0001952189007429665, 'max_depth': 40, 'min_data_in_leaf': 49, 'num_iterations': 518, 'num_leaves': 30, 'objective': 'regression_l2', 'verbose': 10}
# params = {'learning_rate': 0.1, 'max_depth': -1, 'min_data_in_leaf': 20, 'num_iterations': 100, 'num_leaves': 31, 'objective': 'regression_l2', 'verbose': 10}
# params = {'learning_rate': 0.00017211187071323217, 'max_depth': 0, 'min_data_in_leaf': 47, 'num_iterations': 789, 'num_leaves': 40, 'objective': 'regression_l2', 'verbose': 10}
params = {'learning_rate': 0.00015563327393457418, 'max_depth': 20, 'min_data_in_leaf': 49, 'num_iterations': 866, 'num_leaves': 40, 'objective': 'regression_l2', 'verbose': 10}



# Initialize GPBoostRegressor
model = gpb.GPBoostRegressor(**params)

# Fit the model
model.fit(X_train, y_train, gp_model=gp_model)

# predict
pred = model.predict(X_test, group_data_pred=group_test)
y_pred = pd.DataFrame(data=pred['response_mean'], columns=['response_mean'], index=y_test.index)
# evaluate
evaluate_model(y_test, y_pred)

# Calculate the importance of SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.plots.heatmap(shap_values)
plt.show()

