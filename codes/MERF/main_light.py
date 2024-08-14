import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from merf import plot_merf_training_stats
from merf.merf import MERF



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

# Instantiate the LightGBM model
lgb_params = {'learning_rate': 0.009261183930341806, 'max_depth': 20, 'min_data_in_leaf': 48, 'num_iterations': 213, 'num_leaves': 10, 'objective': 'regression_l2', 'verbose': 10}

# 实例化LightGBM模型
lgb_model = lgb.LGBMRegressor(**lgb_params)

# Fit the LightGBM model
merf_model = MERF(
    fixed_effects_model=lgb_model,
    max_iterations=100,
)
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
evaluate_model(y_test.values.flatten(), y_pred)

explainer = shap.Explainer(merf_model.trained_fe_model)
shap_values = explainer(X_test_merf)

shap.summary_plot(shap_values, X_train, plot_type="bar")
plt.savefig('./results/merf_light_importance.jpg')
plt.show()
