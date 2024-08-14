# 导入所需的库
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.metrics import accuracy_score,roc_auc_score


def evaluate_model(y__test, y__pred):
    r2 = r2_score(y__test, y__pred)
    mse = mean_squared_error(y__test, y__pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")


# Read data
X = pd.read_csv(r'.\data\standard_data\data.csv').drop('ID', axis=1)
y = pd.read_csv(r'.\data\standard_data\y.csv').drop('ID', axis=1)

# Divide the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, shuffle=False)

# Convert data to LightGBM's data format
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# train
# model = lgb.LGBMRegressor(num_leaves=127,
#                           min_data_in_leaf=20,
#                           max_depth=-1,
#                           max_bin=255,
#                           objective='mse',
#                           num_threads=1)

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


def objective(params):
    params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
    params['n_estimators'] = int(params['n_estimators'])
    params['num_leaves'] = int(params['num_leaves'])
    # params['max_bin'] = int(params['max_bin'])

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_metric='mse', eval_set=[(X_test, y_test)])

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evaluate_model(y_test, y_pred)
    print("Training with parameters:", params)
    return mse


trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=4000, trials=trials)
best_params = space_eval(space, best)
print("Best parameters:", best_params)

print('best:')
objective(best_params)
'''
Mean Squared Error: 0.5124751033430428
R-squared: 0.4875248966569572
best: 
{'learning_rate': 0.0999308799908438, 'max_depth': 30, 'min_data_in_leaf': 19, 'n_estimators': 350, 'num_leaves': 30, 'num_threads': 2, 'objective': 'regression_l2'}
'''