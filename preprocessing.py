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


def features_separate(df: pd.DataFrame, threshold: int) -> Tuple[List[str], List[str]]:
    """
    Разделяет признаки датафрейма на категориальные и числовые на основе порога уникальных значений.

    Args:
        df (pd.DataFrame): Исходный датафрейм.
        threshold (int): Порог уникальных значений для классификации категориальных признаков.

    Returns:
        Tuple[List[str], List[str]]: Список категориальных и числовых признаков.
    """
    categorical_columns = []
    numerical_columns = []

    for column_name in df.columns:
        if df[column_name].nunique() < threshold:
            categorical_columns.append(column_name)
        else:
            numerical_columns.append(column_name)

    return categorical_columns, numerical_columns


def save_split_descr(df_initial: pd.DataFrame,
                    initial_columns: List[str],
                    target: pd.Series,
                    target_name: str,
                    store: str='data/',
                    dir: str='split_description/'
                    ) -> None:

    FILE_PATH = f'{store}{dir}{target_name}_descr.pkl'

    df = df_initial[initial_columns]

    target_indexes = target.index
    df_description = df.loc[target_indexes, :]

    save_checker = False
    for col_name in initial_columns:
        check_true = target.to_numpy() == df_description[col_name]
        if check_true.all():
            df_description.to_pickle(FILE_PATH)
            save_checker = True

    if save_checker:
        print(f'File {FILE_PATH} saved !')

    else:
        print('Something wrong!')
