import pandas as pd
import numpy as np
from merf import plot_merf_training_stats
from merf.merf import MERF
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt


def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
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
                                            data=shap_value.data[:, [i for i in range(shap_value.shape[1]) if i != remove_index]],
                                            # display_data=shap_value.data[:,remove_index],
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

df = pd.read_csv(r'.\data\standard_data\data.csv').drop('ID', axis=1)
y = pd.read_csv(r'.\data\standard_data\y.csv').drop('ID', axis=1)
CID_train = pd.read_csv(r'./data/order_split/CID_train.csv').drop('ID', axis=1)
CID_test = pd.read_csv(r'./data/order_split/CID_test.csv').drop('ID', axis=1)
Year_train = pd.read_csv(r'./data/order_split/Year_train.csv').drop('ID', axis=1)
Year_test = pd.read_csv(r'./data/order_split/Year_test.csv').drop('ID', axis=1)


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.333, shuffle=False)

clusters_train = X_train['CID']
clusters_test = X_test['CID']

X_train_merf = X_train.drop(['CID'], axis=1)
X_test_merf = X_test.drop(['CID'], axis=1)
X = pd.concat([X_train_merf,X_test_merf], axis=0, ignore_index=True)
Year = pd.concat([Year_train, Year_test], axis=0, ignore_index=True)
CID = pd.concat([CID_train, CID_test], axis=0, ignore_index=True)
label_list = Year['Year'].astype(str) + str('-') + CID['CID'].astype(str)

Z_train = np.ones(shape=(X_train.shape[0], 1))
Z_test = np.ones(shape=(X_test.shape[0], 1))


xgb_params = {'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.028624246810408097, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 101, 'subsample': 0.8}

xgb_model = xgb.XGBRegressor(**xgb_params)

merf_model = MERF(
    fixed_effects_model=xgb_model,
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
# training_stats = plot_merf_training_stats(merf_model, num_clusters_to_plot=10)
# plt.show()


y_pred = merf_model.predict(X_test_merf, Z_test, clusters_test)
evaluate_model(y_test.values.flatten(), y_pred)

#======================explain
explain_data = X.iloc[1163:1225,:]
explain_label = label_list.iloc[1163:1225]
# 使用SHAP解释固定效应模型
explainer = shap.Explainer(merf_model.trained_fe_model, X_train_merf)
shap_values = explainer(explain_data)
filtered_shap_values = filter_shap_value(shap_values, 'Year')
filtered_X = X.drop(columns=['Year'])

# # #===============================重要性绘制
# 绘制SHAP值
# plt.figure(figsize=(12, 20))
# 使用索引过滤掉 "Year" 列
# filtered_features = X.drop(columns=["Year"])
# summary_plot_text(filtered_shap_values, filtered_features,
#                  show=False, name='shap_merf_xg_importance.svg')

# # #===============================heatmap
shap.plots.heatmap(filtered_shap_values,
                   max_display=15,
                   plot_width=60,
                   label_list=explain_label,
                   show=False,
                   )
plt.savefig('merf_heatmap_2018.svg')

# # #==========================聚类
# clustering = shap.utils.hclust(filtered_X, y)
# shap.plots.bar(filtered_shap_values, clustering=clustering, clustering_cutoff=0.7, max_display=37, show=False)
# plt.savefig('bar.svg')
# plt.show()

# # #=======================local
# sample_id = 348
# sample_id = 355
# sample_id = 367
# shap.plots.bar(filtered_shap_values[sample_id], max_display=20, show=False)
# plt.savefig(f'{sample_id}_bar.svg')
# shap.plots.bar(shap_values[sample_id], max_display=20, show=True)
