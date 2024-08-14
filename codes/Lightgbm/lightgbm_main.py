# 导入所需的库
import lightgbm as lgb
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.metrics import accuracy_score,roc_auc_score


def evaluate_model(model, y__test, y__pred):
    r2 = r2_score(y__test, y__pred)
    mse = mean_squared_error(y__test, y__pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

def filter_shap_value(shap_value, remove_name):
    values = shap_value.values
    feature_names = shap_value.feature_names
    feature_to_remove = remove_name
    remove_index = feature_names.index(feature_to_remove)
    filtered_values = values[:, [i for i in range(values.shape[1]) if i != remove_index]]
    filtered_feature_names = [name for i, name in enumerate(feature_names) if i != remove_index]
    filtered_shap_values = shap.Explanation(filtered_values,
                                            base_values=shap_value.base_values,
                                            data=shap_value.data,
                                            feature_names=filtered_feature_names)
    return filtered_shap_values

def summary_plot_text(shap_values, features, show=True, name='importance'):
    name = name
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    ax = plt.gca()
    for bar in ax.patches:
        width = bar.get_width()
        ax.text(bar.get_x() + width + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{width:.4f}', ha='left', va='center')

    plt.xlim(0, ax.get_xlim()[1] * 1.1)
    if show :
        plt.show()
    else:
        plt.savefig(name)

# Read data
X = pd.read_csv(r'.\data\standard_data\data.csv').drop('ID', axis=1)
y = pd.read_csv(r'.\data\standard_data\y.csv').drop('ID', axis=1)

# Divide the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, shuffle=False)

# Convert data to LightGBM's data format
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Train the model
params = {'learning_rate': 0.0999308799908438,
          'max_depth': 30,
          'min_data_in_leaf': 19,
          'n_estimators': 350,
          'num_leaves': 30,
          'num_threads': 2,
          'objective': 'regression_l2'}

# params = {'learning_rate': 0.1,
#           'max_depth': -1,
#           'min_data_in_leaf': 20,
#           'num_iterations': 100,
#           'num_leaves': 31,
#           'objective': 'regression'}

model = lgb.LGBMRegressor(**params)

model.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)])

# predict
y_pred = model.predict(X_test)

# evaluate
evaluate_model(model, y_test, y_pred)

# Calculate the importance of SHAP
explainer = shap.TreeExplainer(model, X_train)
shap_values = explainer(X)
filtered_shap_values = filter_shap_value(filter_shap_value(shap_values, 'Year'), 'CID')
filtered_features = X.drop(columns=["Year", "CID"])
summary_plot_text(filtered_shap_values, filtered_features,
                  show=False, name='shap_light_importance.svg')
plt.show()
