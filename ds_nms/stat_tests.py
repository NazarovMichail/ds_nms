import pandas as pd
from typing import List, Tuple, Any, Dict, Literal, Union
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


def plot_test_hists(sample_1: pd.Series,
                    sample_2: pd.Series) -> None:
    """Строит гистограммы для двух выборок.

    Args:
        sample_1 (pd.Series): Первая выборка.
        sample_2 (pd.Series):  Вторая выборка.
    """
    fig, ax = plt.subplots(1,2, figsize=[10, 5])
    sample_1.hist(ax=ax[0])
    sample_2.hist(ax=ax[1])
    ax[0].set_title('Выборка 1')
    ax[1].set_title('Выборка 2');

def kolmog_smirn_test(sample_1: pd.Series,
                    sample_2: pd.Series,
                    alpha: float=0.05) -> None:
    """Тест на то, принадлежат ли две выборки одному и тому же распределению

    Args:
        sample_1 (pd.Series): Выборка 1
        sample_2 (pd.Series): Выборка 2
        alpha (float, optional): Вероятность ошибочно отклонить нулевую гипотезу. Defaults to 0.05 (уровень значимости).
    """

    statistic, p_value = kstest(sample_1, sample_2)

    if p_value < alpha:
        print(f'p-value={p_value:.5f}')
        print("Гипотеза о равенстве распределения отвергается")
    else:
        print(f'p-value={p_value:.5f}')
        print("Данные могут быть взяты из одного распределения")


def kraskel_wallis_test(sample_1: pd.Series,
                        sample_2: pd.Series,
                        alpha: float=0.05) -> None:
    """Тест для сравнения независимых выборок и определения, есть ли статистически значимые различия между ними

    Args:
        sample_1 (pd.Series): Выборка 1
        sample_2 (pd.Series): Выборка 2
        alpha (float, optional): Вероятность ошибочно отклонить нулевую гипотезу. Defaults to 0.05 (уровень значимости).
    """

    statistic, p_value = kruskal(sample_1, sample_2)
    if p_value < alpha:
        print(f'p-value={p_value:.5f}')
        print("Гипотеза о равенстве распределения отвергается")
    else:
        print(f'p-value={p_value:.5f}')
        print("Данные могут быть взяты из одного распределения")
