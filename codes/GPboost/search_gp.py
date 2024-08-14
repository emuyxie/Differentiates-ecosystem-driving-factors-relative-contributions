import gpboost as gpb
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from hyperopt import hp, fmin, tpe, Trials, space_eval
import matplotlib.pyplot as plt


def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")


# Define the objective function for hyperparameter optimization
def objective(params):
    gp_model = gpb.GPModel(group_data=group_train)
    data_train = gpb.Dataset(X_train, y_train)
    bst = gpb.train(params=params, train_set=data_train, gp_model=gp_model)
    gp_model.summary()
    pred = bst.predict(data=X_test, group_data_pred=group_test)
    y_pred = pd.DataFrame(data=pred['response_mean'], columns=['response_mean'])
    evaluate_model(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("Training with parameters:", params)
    return mse




# Define the search space for hyperparameters
space = {
    'num_iterations': hp.choice('num_iterations', range(100, 1000)),
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.1),
    'max_depth': hp.choice('max_depth', [-1, 0, 10, 20, 30, 40]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', range(10, 50)),
    'num_leaves': hp.choice('num_leaves', [10, 20, 30, 40]),
    'objective': 'regression_l2',
    'verbose': 10
}

# Read data
# df = pd.read_csv('E:/Programs/Neimeng/Py/data/original.csv')
X_train = pd.read_csv('./data/order_split/X_train_df.csv').drop('ID', axis=1)
X_test = pd.read_csv('./data/order_split/X_test_df.csv').drop('ID', axis=1)
y_train = pd.read_csv('./data/order_split/y_train_df.csv').drop('ID', axis=1)
y_test = pd.read_csv('./data/order_split/y_test_df.csv').drop('ID', axis=1)
CID_train = pd.read_csv('./data/order_split/CID_train.csv').drop('ID', axis=1)
CID_test = pd.read_csv('./data/order_split/CID_test.csv').drop('ID', axis=1)
Year_train = pd.read_csv('./data/order_split/Year_train.csv').drop('ID', axis=1)
Year_test = pd.read_csv('./data/order_split/Year_test.csv').drop('ID', axis=1)

# Prepare group data
group_train = pd.concat([CID_train], axis=1)
group_test = pd.concat([CID_test], axis=1)

# Perform hyperparameter optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=2000, trials=trials)

results = trials.results
losses = [r['loss'] for r in results]

plt.figure(figsize=(8, 6))
plt.plot(losses, marker='o', linestyle='-', color='b')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Loss (MSE)', fontsize=14)
plt.title('Hyperparameter Optimization Progress', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the best parameters found


print("Best parameters found:")
# Evaluate the model with the best parameters
best_params = space_eval(space, best)
objective(best_params)

# Evaluate the model
print(best)
print(best_params)

'''
Mean Squared Error: 0.11387061210740371
R-squared: 0.8861293878925963
{'learning_rate': 0.00015563327393457418, 'max_depth': 20, 'min_data_in_leaf': 49, 'num_iterations': 866, 'num_leaves': 40, 'objective': 'regression_l2', 'verbose': 10}
'''