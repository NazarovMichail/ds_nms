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
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, median_absolute_error, root_mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import mlflow
from mlflow.models import infer_signature
from permetrics.regression import RegressionMetric
import shap
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit


def get_feature_importance_df(X: pd.DataFrame, model: BaseEstimator) -> pd.DataFrame:
    """Формирует датафрейм с важностью (весами) признаков для обученной модели.

    Args:
        X (pd.DataFrame): Исходный датафрейм
        model (BaseEstimator): Обученная модель

    Returns:
        pd.DataFrame: Датафрейм с двумя столбцами:
         - 'features' (название признака)
         - 'importances' (вес признака или показатель важности)
    """
    try:
        # Если у модели есть .coef_ (линейные модели)
        sorted_inds = np.abs(model.coef_).argsort()[::-1]
        colnames = X.columns[sorted_inds]
        coefs = model.coef_[sorted_inds]
        feature_importances_df = pd.DataFrame({
            'features': colnames,
            'importances': coefs.round(3)
        })
    except AttributeError:
        # Если у модели есть .feature_importances_ (деревья, ансамбли)
        importances = model.feature_importances_
        feature_importances_df = (
            pd.DataFrame({
                'features': X.columns,
                'importances': importances
            })
            .sort_values(by='importances', ascending=False)
            .reset_index(drop=True)
        )

    return feature_importances_df

def relative_error(y_pred: np.array,
            y_true: pd.Series,
            re_threshold: int=30) -> int:
    """Вычисляет долю объектов с относительной ошибкой меньше порогового значения

    Args:
        y_pred (np.array): Вектор предсказанных целевых переменных
        y_true (pd.Series): Вектор действительных целевых переменных
        re_threshold (int, optional): Пороовое значение относительной ошибки. Defaults to 30.

    Returns:
        int: Доля объектов, для которых значения относительной ошибки меньше порогового значения
    """

    re = abs(y_pred - y_true)
    re_relative = (re / (y_pred + 1e-8)) * 100
    count_less_thresh = re_relative[re_relative < re_threshold].shape[0]
    count_less_thresh_ratio = count_less_thresh / y_true.shape[0]

    return count_less_thresh_ratio

def get_prediction(
    X: pd.DataFrame,
    y: pd.Series,
    model: BaseEstimator,
    re_threshold: int = 30
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Возвращает датафрейм с метриками и массив предсказаний модели.

    Args:
        X (pd.DataFrame): Датафрейм для предсказания
        y (pd.Series): Целевая переменная
        model (BaseEstimator): Обученная модель
        re_threshold (int, optional): Пороовое значение относительной ошибки. Defaults to 30.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Кортеж датафрейма с метриками и массив предсказаний модели
    """
    y_pred = model.predict(X)

    RMSE = root_mean_squared_error(y, y_pred)
    MAE = mean_absolute_error(y, y_pred)
    RE = relative_error(y_pred=y_pred,
                        y_true=y,
                        re_threshold=re_threshold)
    negative = (y_pred < 0).sum()

    metrics_dict = {
        "RMSE": RMSE,
        "MAE": MAE,
        "RE": RE,
        "negative": negative,
    }

    # Проверяем, что в выборке больше 1 точки (иначе r2_score не посчитать)
    if y_pred.shape[0] > 1:
        R2 = r2_score(y, y_pred)
        # Чтобы избежать деления на ноль:
        denom = (y.max() - y.min())
        if denom == 0:
            NRMSE = np.nan
        else:
            NRMSE = RMSE / denom
        metrics_dict["R2"] = R2
        metrics_dict["NRMSE"] = NRMSE

    metrics_df = pd.DataFrame(metrics_dict, index=[0])
    return metrics_df, y_pred

def train_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model: BaseEstimator,
    cv_type: Literal['kf', 'loo', 'stratify', 'ts'],
    metric_best: Literal['R2_val', 'RMSE_val', 'NRMSE_val', 'MAE_val', 'RE_val' ],
    stratify: Union[str, None] = 'quantile',  # только для 'stratify'
    n_splits: int = 5,                       # KFold / StratifiedKFold
    shuffle: bool = False,                   # KFold / StratifiedKFold
    train_size: int = 48,                    # TimeSeriesSplit
    val_size: int = 12,                      # TimeSeriesSplit
    data_name: Union[str, None] = None
) -> Tuple[BaseEstimator, dict]:
    """
    Единая функция для обучения модели с разными схемами кросс-валидации:
      - 'stratify' : StratifiedKFold (доп. стратификация по квантилям или столбцу)
      - 'loo'      : LeaveOneOut
      - 'kf'       : KFold
      - 'ts'       : TimeSeriesSplit

    Параметры
    ---------
    X : pd.DataFrame
        Матрица признаков
    y : pd.Series
        Целевые значения
    model : BaseEstimator
        Модель (sklearn), у которой есть .fit/.predict
    cv_type : {'stratify', 'loo', 'kf', 'ts'}
        Какую схему кросс-валидации использовать
    stratify : str или None
        - 'quantile': стратификация по квартилям y (4 корзины)
        - иначе: название столбца X для стратификации
    n_splits : int
        Количество сплитов (KFold / StratifiedKFold)
    shuffle : bool
        Перемешивать ли данные в KFold / StratifiedKFold
    train_size : int
        Размер обучающей выборки для TimeSeriesSplit
    val_size : int
        Размер валидации (окна) для TimeSeriesSplit
    data_name : str или None
        Название набора данных (для логов)

    Возвращает
    ---------
    best_model : BaseEstimator
        Обученная модель (либо «лучшая» — в Stratified/KFold/LOO,
                          либо последняя — в TS)
    final_result : dict
        Словарь со статистикой метрик и сами метрики по сплитам
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Словарь для хранения всех метрик train/val (включая "negative")
    results_dict = {
        'R2_train': [],
        'RMSE_train': [],
        'NRMSE_train': [],
        'MAE_train': [],
        'RE_train': [],
        'negative_train': [],

        'R2_val': [],
        'RMSE_val': [],
        'NRMSE_val': [],
        'MAE_val': [],
        'RE_val': [],
        'negative_val': []
    }
    models_history = []
    y_val_full = []
    y_pred_full = []

    # -------------------------- #
    # 1. Определяем способ split
    # -------------------------- #
    if cv_type == 'stratify':
        # Подготовка страты
        quant_25 = y.quantile(0.25)
        quant_50 = y.quantile(0.50)
        quant_75 = y.quantile(0.75)

        if stratify == 'quantile':
            def _strat_func(val: float) -> int:
                if val <= quant_25:
                    return 1
                elif val <= quant_50:
                    return 2
                elif val <= quant_75:
                    return 3
                else:
                    return 4
            strat_vector = y.apply(_strat_func)
        else:
            strat_vector = X[stratify]

        cv_splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
        )
        split_iter = cv_splitter.split(X, strat_vector)
        desc_text = f"StratifiedKFold (n_splits={n_splits})"

    elif cv_type == 'loo':
        cv_splitter = LeaveOneOut()
        split_iter = cv_splitter.split(X)
        desc_text = "LeaveOneOut"

    elif cv_type == 'kf':
        cv_splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle
        )
        split_iter = cv_splitter.split(X)
        desc_text = f"KFold (n_splits={n_splits})"

    elif cv_type == 'ts':
        n_splits_ts = (len(X) - train_size) // val_size
        cv_splitter = TimeSeriesSplit(
            n_splits=n_splits_ts,
            test_size=val_size
        )
        split_iter = cv_splitter.split(X)
        desc_text = f"TimeSeriesSplit (n_splits={n_splits_ts})"
    else:
        raise ValueError(
            f"Неверный cv_type='{cv_type}'. "
            f"Допустимые: 'stratify', 'loo', 'kf', 'ts'."
        )

    # -------------------------- #
    # 2. Основной цикл
    # -------------------------- #
    progrbar = tqdm(split_iter, total=cv_splitter.get_n_splits(X), desc=desc_text)
    for train_idx, val_idx in progrbar:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)

        # Получаем метрики на train
        metrics_train_df, y_train_pred = get_prediction(X_train, y_train, model)
        # Получаем метрики на val
        metrics_val_df, y_val_pred = get_prediction(X_val, y_val, model)

        R2_train = metrics_train_df.get("R2", np.nan)
        RMSE_train = metrics_train_df["RMSE"]
        NRMSE_train = metrics_train_df.get("NRMSE", np.nan)
        MAE_train = metrics_train_df["MAE"]
        RE_train = metrics_train_df["RE"]
        negative_train = metrics_train_df["negative"]

        R2_val = metrics_val_df.get("R2", np.nan)
        RMSE_val = metrics_val_df["RMSE"]
        NRMSE_val = metrics_val_df.get("NRMSE", np.nan)
        MAE_val = metrics_val_df["MAE"]
        RE_val = metrics_val_df["RE"]
        negative_val = metrics_val_df["negative"]

        # Заполняем общий словарь
        results_dict['R2_train'].append(R2_train)
        results_dict['RMSE_train'].append(RMSE_train)
        results_dict['NRMSE_train'].append(NRMSE_train)
        results_dict['MAE_train'].append(MAE_train)
        results_dict['RE_train'].append(RE_train)
        results_dict['negative_train'].append(negative_train)

        results_dict['R2_val'].append(R2_val)
        results_dict['RMSE_val'].append(RMSE_val)
        results_dict['NRMSE_val'].append(NRMSE_val)
        results_dict['MAE_val'].append(MAE_val)
        results_dict['RE_val'].append(RE_val)
        results_dict['negative_val'].append(negative_val)

        models_history.append(model)
        y_val_full.append(y_val)
        y_pred_full.append(y_val_pred)

    # -------------------------- #
    # 3. Преобразуем и ищем «лучшую» модель
    # -------------------------- #
    for k, arr in results_dict.items():
        results_dict[k] = np.array(arr)

    if cv_type in ('ts'):
        # Для 'ts' — последняя
        best_model = models_history[-1]

    if cv_type == "loo":
        # Логика выбора "лучшей": к примеру, минимизируем MAE_val
        if metric_best in ('RMSE_val', 'MAE_val', 'RE_val' ):
            if metric_best in ('RMSE_val', 'MAE_val'):
                best_score_ind = results_dict[metric_best].argmin()
                best_model = models_history[best_score_ind]
            else:
                best_score_ind = results_dict[metric_best].argmax()
                best_model = models_history[best_score_ind]
        else:
            raise ValueError("""Для LOO метрика выбора лучшей модели (metric_best)
                должна быть: RMSE_val, MAE_val, RE_val""")
    else:
        if metric_best in ('RMSE_val', 'MAE_val'):
            best_score_ind = results_dict[metric_best].argmin()
            best_model = models_history[best_score_ind]
        else:
            best_score_ind = results_dict[metric_best].argmax()
            best_model = models_history[best_score_ind]
    # -------------------------- #
    # 4. Сводный словарь с результатами
    # -------------------------- #
    final_result = {
        'model': str(model),
        'data_name': data_name
    }

    for key, values in results_dict.items():
        final_result[f'{key}_macro'] = np.nanmean(values).round(3)
        final_result[f'{key}_std'] = np.nanstd(values).round(3)
        final_result[f'{key}_splits'] = np.round(values, 3)

    y_val_concat = pd.concat(y_val_full).values
    y_pred_concat = np.concatenate(y_pred_full)
    if len(y_val_concat) > 1:
        final_result['R2_val_micro'] = r2_score(y_val_concat, y_pred_concat)
        final_result['RMSE_val_micro'] = root_mean_squared_error(
            y_val_concat, y_pred_concat
        )
        final_result['MAE_val_micro'] = mean_absolute_error(
            y_val_concat, y_pred_concat
        )
        final_result['RE_val_micro'] = relative_error(y_pred=y_pred_concat,
                               y_true=y_val_concat)
        final_result['Negative_micro'] = (y_pred_concat < 0).sum()
        val_denom = (y_val_concat.max() - y_val_concat.min())
        if val_denom == 0:
            final_result['NRMSE_val_micro'] = np.nan
        else:
            final_result['NRMSE_val_micro'] = (
                final_result['RMSE_val_micro'] / val_denom
            )
    else:
        final_result['R2_val_micro'] = np.nan
        final_result['RMSE_val_micro'] = np.nan
        final_result['MAE_val_micro'] = np.nan
        final_result['RE_val_micro'] = np.nan
        final_result['NRMSE_val_micro'] = np.nan

    # Метрики разности валидационных значений и обучающих
    if cv_type == 'loo':
        R2_diff = abs(final_result['R2_train_macro'] - final_result['R2_val_micro'])
        RMSE_diff = abs(final_result['RMSE_train_macro'] - final_result['RMSE_val_micro'])
        MAE_diff = abs(final_result['MAE_train_macro'] - final_result['MAE_val_micro'])
    else:
        R2_diff = (final_result['R2_train_splits'] - final_result['R2_val_splits']).mean()
        RMSE_diff = abs(final_result['RMSE_train_splits'] - final_result['RMSE_val_splits']).mean()
        MAE_diff = abs(final_result['MAE_train_splits'] - final_result['MAE_val_splits']).mean()

    final_result['R2_diff_rel'] = R2_diff / final_result['R2_train_macro']
    final_result['RMSE_diff_rel'] = RMSE_diff / final_result['RMSE_train_macro']
    final_result['MAE_diff_rel'] = MAE_diff / final_result['MAE_train_macro']

    clear_output()

    return best_model, final_result
