import pandas as pd
import numpy as np
from merf import plot_merf_training_stats
from merf.merf import MERF
import gpboost as gpb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt


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

# build GPModel
# gp_model = gpb.GPModel(group_data=clusters_train)

# Define parameters
params = {'learning_rate': 0.011082706950120428, 'max_depth': -1, 'min_data_in_leaf': 26, 'num_iterations': 156, 'num_leaves': 10, 'objective': 'regression_l2', 'verbose': 10}


# Initialize GPBoostRegressor
gpboost_model = gpb.GPBoostRegressor(**params)


# Instantiate the MERF model, set the fixed_effects_model parameters to the GPBoost model, and use the provided parameters
merf_model = MERF(
    fixed_effects_model=gpboost_model,
    max_iterations=100,
)

# Fit the MERF model
merf_model.fit(X_train_merf,
               Z_train,
               clusters_train,
               y_train.values.flatten(),
               X_test_merf,
               Z_test,
               clusters_test,
               y_test.values.flatten())

training_stats = plot_merf_training_stats(merf_model, num_clusters_to_plot=10)
plt.show()

# predict
y_pred = merf_model.predict(X_test_merf, Z_test, clusters_test)
evaluate_model(y_test, y_pred)

# Use SHAP to calculate feature importance
explainer = shap.Explainer(merf_model.trained_fe_model)
shap_values = explainer(X_test_merf)

shap.summary_plot(shap_values, X_train, plot_type="bar")
plt.savefig('./results/merf_gp_importance.jpg')
plt.show()
