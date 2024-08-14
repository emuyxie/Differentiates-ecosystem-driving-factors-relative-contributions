import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def evaluate_model(y__test, y__pred):
    r2 = r2_score(y__test, y__pred)
    mse = mean_squared_error(y__test, y__pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")



X = pd.read_csv(r'.\data\standard_data\data.csv').drop('ID', axis=1)
y = pd.read_csv(r'.\data\standard_data\y.csv').drop('ID', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.333,
                                                    shuffle=False)


dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)
dtest = xgb.DMatrix(X_test, label=y_test, nthread=-1)


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



def objective(params):
    # params['max_depth'] = int(params['max_depth'])
    # params['n_estimators'] = int(params['n_estimators'])
    params['objective'] = 'reg:squarederror'
    params['eval_metric'] = 'rmse'

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evaluate_model(y_test.values.flatten(), y_pred)
    print("Training with parameters:", params)
    return mse



trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=10000, trials=trials)
best_params = space_eval(space, best)
print("Best parameters:", best_params)



print('\nbest:\n')

objective(best_params)


'''
Mean Squared Error: 0.4600100898636132
R-squared: 0.5399899101363868
Training with parameters: {'colsample_bytree': 0.5, 'learning_rate': 0.08380595562735904, 'max_depth': 14, 'min_child_weight': 2, 'n_estimators': 960, 'subsample': 0.6, 'objective': 'reg:squarederror', 'eval_metric': 'rmse'}

Training with parameters: {'colsample_bytree': 0.5, 'gamma': 0, 'learning_rate': 0.09037587083370761, 'max_depth': 13, 'min_child_weight': 1, 'n_estimators': 150, 'subsample': 0.6, 'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
R-squared: 0.5746503853426812
Mean Squared Error: 0.42534961465731874

'''


