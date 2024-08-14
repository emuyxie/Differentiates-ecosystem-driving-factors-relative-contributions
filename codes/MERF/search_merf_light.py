from merf.merf import MERF
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, space_eval
import matplotlib.pyplot as plt

# 创建示例数据
def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

df = pd.read_csv(r'.\data\standard_data\data.csv').drop('ID', axis=1)
y = pd.read_csv(r'.\data\standard_data\y.csv').drop('ID', axis=1)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.333, shuffle=False)

clusters_train = X_train['CID']
clusters_test = X_test['CID']

X_train_merf = X_train.drop(['CID'], axis=1)
X_test_merf = X_test.drop(['CID'], axis=1)

Z_train = np.ones(shape=(X_train.shape[0], 1))
Z_test = np.ones(shape=(X_test.shape[0], 1))

# Define the objective function for hyperparameter optimization
def objective(params):
    params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
    params['n_estimators'] = int(params['n_estimators'])
    params['num_leaves'] = int(params['num_leaves'])
    model = lgb.LGBMRegressor(**params)
    merf_model = MERF(
        fixed_effects_model=model,
        max_iterations=30,
    )
    merf_model.fit(X_train_merf, Z_train, clusters_train, y_train.values.flatten())
    y_pred = merf_model.predict(X_test_merf, Z_test, clusters_test)
    mse = mean_squared_error(y_test.values.flatten(), y_pred)
    evaluate_model(y_test.values.flatten(), y_pred)

    return mse

# Define the search space for hyperparameters
space = {
    'objective': 'regression_l2',
    'n_estimators': hp.choice('num_iterations', [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350]),
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.1),
    'num_leaves': hp.choice('num_leaves', [10, 20, 30, 40]),
    'num_threads': 2,
    'max_depth': hp.choice('max_depth', [5, 10, 15, 20, 25, 30, 35, 40]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', range(10, 50)),
    # 'max_bin': hp.choice('max_bin', [5, 10, 15, 20]),
}

# Perform hyperparameter optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=500, trials=trials)

best_params = space_eval(space, best)
# Print the best parameters found
print("Best parameters found:")
print(best_params)
print('best:')
objective(best_params)

'''
{'learning_rate': 0.009261183930341806, 'max_depth': 20, 'min_data_in_leaf': 48, 'num_iterations': 213, 'num_leaves': 10, 'objective': 'regression_l2', 'verbose': 10}
Mean Squared Error: 0.12075777250412836
R-squared: 0.8792422274958717
'''
