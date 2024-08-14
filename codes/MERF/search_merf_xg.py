from merf.merf import MERF
import xgboost as xgb
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
    params['objective'] = 'reg:squarederror'
    params['eval_metric'] = 'rmse'
    model = xgb.XGBRegressor(**params)
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
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
    'max_depth': hp.choice('max_depth', range(5, 7)),
    'n_estimators': hp.choice('n_estimators', range(90, 110)),
    'colsample_bytree': hp.choice('colsample_bytree', [0.5, 1]),
    'min_child_weight': hp.choice('min_child_weight', [1, 2]),
    'subsample': hp.choice('subsample', [0.8, 1]),
    # 'objective': 'reg:squarederror',
    # 'reg_lambda': hp.choice('reg_lambda', [0, 0.001, 0.01, 0.1, 1.0]),
    'gamma': hp.choice('gamma', [0, 0.05, 0.1, 1]),
    # 'device': 'cuda'
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
{'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.028624246810408097, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 101, 'subsample': 0.8}
Mean Squared Error: 0.11332887053583236
R-squared: 0.8866711294641676
'''
