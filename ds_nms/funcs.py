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
