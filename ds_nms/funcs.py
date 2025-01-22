import pandas as pd
from typing import List, Tuple, Any, Dict, Literal
import pickle
import os
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
import optuna
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PassiveAggressiveRegressor, LassoLars, BayesianRidge, HuberRegressor, QuantileRegressor, RANSACRegressor, TheilSenRegressor, PoissonRegressor, TweedieRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate, StratifiedKFold, LeaveOneOut
from tqdm import tqdm
from IPython.display import clear_output
from  datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, kruskal
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer, MinMaxScaler, PowerTransformer, TargetEncoder, PolynomialFeatures
from IPython.display import display
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, median_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import mlflow
from mlflow.models import infer_signature
from permetrics.regression import RegressionMetric
import shap
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit


def features_sum(df: pd.DataFrame,
                column_names_lst: List[str],
                new_column_name: str) -> pd.DataFrame:

    column_inx = df.columns.get_loc(column_names_lst[0])

    df[new_column_name] = df.loc[:, column_names_lst].sum(axis=1)
    df = df.drop(columns=column_names_lst)

    df_columns = list(df.columns[:-1])
    df_columns.insert(column_inx, new_column_name)
    df = df.reindex(columns=df_columns)

    return df


def df_encoding(
            df_cat: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
            encoder: TargetEncoder,
            scaler: StandardScaler | MinMaxScaler
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    cat_columns = list(df_cat.columns)
    df_train_inx = y_train.index
    df_test_inx = y_test.index

    df_train_cat = df_cat.loc[df_train_inx, :]
    df_test_cat = df_cat.loc[df_test_inx, :]

    train_encoder = encoder
    train_cat_arr = train_encoder.fit_transform(df_train_cat, y_train)
    test_cat_arr = train_encoder.transform(df_test_cat)

    df_train_cat_encoded = pd.DataFrame(train_cat_arr, columns=cat_columns, index=df_train_inx)
    df_test_cat_encoded = pd.DataFrame(test_cat_arr, columns=cat_columns, index=df_test_inx)

    if scaler is not None:
        df_train_cat_encoded, df_test_cat_encoded = df_scaling(df_train=df_train_cat_encoded,
                                                               df_test=df_test_cat_encoded,
                                                               numerical_columns=cat_columns,
                                                               scaler=scaler)
        return df_train_cat_encoded, df_test_cat_encoded

    print(df_train_cat_encoded.shape, df_test_cat_encoded.shape)
    display(df_train_cat_encoded.describe().round(1))
    display(df_test_cat_encoded.describe().round(1))

    return df_train_cat_encoded, df_test_cat_encoded



def get_feature_explain(X_train: pd.DataFrame,
                        trained_model: BaseEstimator,
                        waterfall_dict: Dict[int, str] = None,
                        img_path: str = "Model"
                        ) -> None:

    try:
        os.makedirs(f'data/FE_plots/{img_path}', exist_ok=False)
    except FileExistsError:
        print("Dir exists |")
        print("_"*11)

    explainer = shap.Explainer(trained_model.predict, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, show=False)
    plt.savefig(f"data/FE_plots/{img_path}/FE_{img_path}.jpg", bbox_inches='tight')
    plt.close()
    shap.plots.bar(shap_values, show=False, max_display=None)
    plt.savefig(f"data/FE_plots/{img_path}/FE_bar_{img_path}.jpg", bbox_inches='tight')
    plt.close()

    if waterfall_dict is not None:
        df_inx_reset = X_train.copy()
        df_inx_reset = X_train.reset_index()

        for df_inx, name in waterfall_dict.items():
            print(name)
            shap_inx = df_inx_reset[df_inx_reset['index'] == df_inx].index[0]
            shap.waterfall_plot(shap_values[shap_inx], show=False)
            plt.title(name)
            plt.savefig(f"data/FE_plots/{img_path}/{name}.png", bbox_inches='tight')
            plt.close()

def concat_metrics(result_df: pd.DataFrame, inx: int, data_names: List[str], ) -> pd.DataFrame:

    df_concat = pd.concat([result_df[data_names[0]][inx]])
    for data_name in data_names[1:]:
        df_concat = pd.concat((df_concat, result_df[data_name][inx]))

    df_concat['data_name'] = data_names
    return df_concat




def get_transform_feature(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        column: str, func: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df_trans_train = X_train.copy()
    df_trans_test = X_test.copy()

    fig, ax = plt.subplots(2, 2, figsize=(15,10))

    column_train_trans = func(X_train[column].values)
    column_test_trans = func(X_test[column].values)

    sns.histplot(df_trans_train[column], ax=ax[0,0], bins=30)
    ax[0,0].set_title("Train original")
    ax[0,1].set_title("Train transformed")
    ax[1,0].set_title("Test original")
    ax[1,1].set_title("Test transformed")
    sns.histplot(column_train_trans, ax=ax[0,1], bins=30, color='red')
    sns.histplot(df_trans_test[column], ax=ax[1,0], bins=30)
    sns.histplot(column_test_trans, ax=ax[1, 1], bins=30, color='red')

    df_trans_train[column] = column_train_trans
    df_trans_test[column] = column_test_trans

    return df_trans_train, df_trans_test






def plot_ts_pred(y_train: pd.Series,
                y_pred_train: pd. Series,
                y_test: pd.Series,
                y_pred_test: pd.Series,
                show_all: bool=False,
                confidences=None
                ):

    plt.figure(figsize=(25, 20))
    plt.plot(y_train, label='Тренировочные данные', marker='.')
    plt.plot(y_test, label='Действительные данные (2024)', color='green', marker='.')
    plt.plot(y_pred_test, label='Прогноз (2024)', color='red', linestyle='--', marker='.')

    if show_all:
        plt.plot(y_pred_train, label='Прогноз тренировочных данных', color='orange', linestyle='--', marker='.')
    if confidences:
        plt.fill_between(
            y_test.index,
            confidences[:, 0],
            confidences[:, 1],
            color='red',
            alpha=0.2,
            label='Доверительный интервал',
        )
    plt.title('Прогноз среднемесячного пассажиропотока (val_metro) на 2024 год')
    plt.xlabel('Дата')
    plt.ylabel('Пассажиропоток')
    plt.legend()
    plt.grid()
    plt.show()
