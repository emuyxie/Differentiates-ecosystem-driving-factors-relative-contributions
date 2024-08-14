import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt


def evaluate_model(y__test, y__pred):
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


df = pd.read_csv(r'E:\Programs\Neimeng\Py\GPboost\data\standard_data\data.csv').drop('ID', axis=1)
y = pd.read_csv(r'E:\Programs\Neimeng\Py\GPboost\data\standard_data\y.csv').drop('ID', axis=1)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, shuffle=False)

evals_result = {}

params = {'colsample_bytree': 0.5,
          'gamma': 0,
          'learning_rate': 0.1,
          'max_depth': 6,
          'min_child_weight': 1,
          'n_estimators': 100,
          'subsample': 1,
          'objective': 'reg:squarederror',
          'eval_metric': 'rmse'}


model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, squared=False)
evaluate_model(y_test, y_pred)

explainer = shap.TreeExplainer(model, X_train)
shap_values = explainer(df)
filtered_shap_values = filter_shap_value(filter_shap_value(shap_values, 'Year'), 'CID')
filtered_features = df.drop(columns=["Year", "CID"])
summary_plot_text(filtered_shap_values, filtered_features,
                  show=False, name='shap_xg_importance.svg')
# # # plt.savefig('./results/shap.jpg')
plt.show()

# explainer = shap.TreeExplainer(model)
#
# shap_values = explainer.shap_values(X_train)
# shap.summary_plot(shap_values, X_train)
# # plt.savefig('./results/shap.jpg')
# plt.show()
#
# shap.summary_plot(shap_values, X_train, plot_type="bar")
# # plt.savefig('./results/importance.jpg')
# plt.show()

