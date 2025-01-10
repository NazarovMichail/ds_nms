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


def get_duplicated_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Возвращает датафрейм с дубликатами в указанном столбце.

    Args:
        df (pd.DataFrame): Исходный датафрейм
        column (str): Имя столбца для поиска дубликатов.

    Returns:
        pd.DataFrame: Датафрейм с дубликатами, отсортированный по указанному столбцу.
    """
    duplicate_mask = df[column].duplicated(keep=False)
    result_df = df[duplicate_mask].sort_values(by=column)

    return result_df

def features_separate(df: pd.DataFrame, threshold: int) -> Tuple[List[str], List[str]]:
    """
    Разделяет признаки датафрейма на категориальные и числовые на основе порога уникальных значений.

    Args:
        df (pd.DataFrame): Исходный датафрейм.
        threshold (int): Порог уникальных значений для классификации категориальных признаков.

    Returns:
        Tuple[List[str], List[str]]: Кортеж списков категориальных и числовых признаков.
    """
    categorical_columns = []
    numerical_columns = []

    for column_name in df.columns:
        if df[column_name].nunique() < threshold:
            categorical_columns.append(column_name)
        else:
            numerical_columns.append(column_name)

    return categorical_columns, numerical_columns

def get_stratified_df(X: pd.DataFrame, feature: pd.Series,
                        col_name: str="strat",
                        display_info: bool=True) -> pd.DataFrame:
    """_summary_

    Args:
        X (pd.DataFrame): Исходный датафрейм.
        feature (pd.Series): Переменная для стратификации.
        col_name (str, optional): Имя столбца для уровней стратификации. Defaults to "strat".
        display_info (bool, optional): Отображать информацию о стратификации. Defaults to True.

    Returns:
        pd.DataFrame: Датафрейм с добавленным столбцом стратификации.
    """
    q1 = feature.quantile(0.25)
    q2 = feature.quantile(0.50)
    q3 = feature.quantile(0.75)

    df_stratify = X.copy()
    df_stratify[col_name] = feature.apply(
        lambda x: 0 if x <= q1 else 1 if x <= q2 else 2 if x <= q3 else 3
    )

    if display_info:
        display(df_stratify.head(3))
        display(df_stratify[col_name].value_counts(normalize=True))

    return df_stratify

def save_split_description(df_initial: pd.DataFrame,
                    initial_columns: List[str],
                    target: pd.Series,
                    df_name: str,
                    store: str='data',
                    ) -> None:
    """Сохраняет датафрейм с выбранными индексами целевой переменной и выбранными столбцами

    Args:
        df_initial (pd.DataFrame): Исходный датафрейм
        initial_columns (List[str]): Столбцы для нового датафрейма
        target (pd.Series): Целевая переменная
        df_name (str): Имя нового датафрейма
        store (str, optional): Директория для сохранения нового датафрейма. Defaults to 'data'.

    Raises:
        ValueError: Ошибка, если колонки отсутсвуют в исходном датафрейме
    """
    missing_columns = set(initial_columns) - set(df_initial.columns)
    if missing_columns:
        raise ValueError(f"Следующие колонки отсутствуют в датафрейме: {missing_columns}")

    os.makedirs(store, exist_ok=True)
    file_path = os.path.join(store, f"{df_name}_descr.pkl")

    df_description = df_initial.loc[target.index, initial_columns]
    df_description.to_pickle(file_path)

    print(f"Файл {file_path} сохранен!")


def df_scaling(df_train: pd.DataFrame,
                df_test: pd.DataFrame,
                numerical_columns: List[str],
                scaler: StandardScaler | MinMaxScaler | Normalizer | RobustScaler,
                return_scaler: bool = False
                ) -> Union[Tuple[pd.DataFrame, pd.DataFrame],
                            Tuple[pd.DataFrame, pd.DataFrame, Union[StandardScaler, MinMaxScaler, Normalizer, RobustScaler]]]:
    """Масштабирование числовых данных в тренировочном и тестовом наборах.

    Args:
        df_train (pd.DataFrame): Датафрейм с тренировочными данными.
        df_test (pd.DataFrame): Датафрейм с тестовыми данными.
        numerical_columns (List[str]):  Список числовых колонок для масштабирования.
        scaler (StandardScaler | MinMaxScaler | Normalizer | RobustScaler): Объект стандартизатора.
        return_scaler (bool, optional): Если True, возвращает scaler. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: _description_
    """
    df_train_inx = df_train.index
    df_test_inx = df_test.index

    df_train_num = df_train[numerical_columns]
    df_test_num = df_test[numerical_columns]

    train_scaler = scaler
    train_scaler.fit(df_train_num)

    array_train_num_scaled = train_scaler.transform(df_train_num)
    array_test_num_scaled = train_scaler.transform(df_test_num)

    df_train_num_scaled = pd.DataFrame(array_train_num_scaled, columns=numerical_columns, index=df_train_inx)
    df_test_num_scaled = pd.DataFrame(array_test_num_scaled, columns=numerical_columns, index=df_test_inx)

    display(df_train_num_scaled.describe().round(1))
    display(df_test_num_scaled.describe().round(1))

    if return_scaler:
        return df_train_num_scaled, df_test_num_scaled, train_scaler

    return df_train_num_scaled, df_test_num_scaled


def drop_outliers_iso(X: pd.DataFrame, y: pd.Series,
                contamination: float=0.04,
                n_estimators: int=100) -> Tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame]:
    """Удаляет выбросы из данных с использованием IsolationForest.

    Args:
        X (pd.DataFrame): Датафрейм с данными
        y (pd.Series): Целевая переменная
        contamination (float, optional): Доля выбросов в данных. Defaults to 0.04.
        n_estimators (int, optional): Количество деревьев в IsolationForest. Defaults to 100.

    Returns:
        Tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame]: X без выбросов, y без выбросов, датафрейм выбросов
    """
    irf = IsolationForest(contamination=contamination,
                    n_estimators=n_estimators,
                    random_state=1)
    irf.fit(X)
    prediction = irf.predict(X)

    clear_mask = prediction == 1
    outlier_mask = prediction == -1

    X_cleared = X[clear_mask]
    y_cleared = y[clear_mask]
    outliers = X[outlier_mask]

    print(f"Удалено {outlier_mask.sum()} объектов из {X.shape[0]}")

    return X_cleared, y_cleared, outliers


def drop_outliers_tuk(X: pd.DataFrame, y: pd.Series,
                feature : str,
                left:float=1.5, right:float=1.5,
                log_scale:bool=False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Удаляет выбросы из признаков и целевой переменной с использованием метода Тьюки

    Args:
        X (pd.DataFrame): Датафрейм с данными
        y (pd.Series): Целевая переменная
        feature (str): Имя столбца, по которому вычисляются выбросы.
        left (float, optional): Коэффициент для нижней границы. Defaults to 1.5.
        right (float, optional): Коэффициент для верхней границы. Defaults to 1.5.
        log_scale (bool, optional): Применить логарифмирование к данным перед вычислением. Defaults to False.

    Raises:
        ValueError: ошибка, если в данных есть отрицательные значения (логарифмирование невозможно)

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]: X без выбросов, y без выбросов, датафрейм выбросов
    """
    if log_scale:
        if (X[feature] <= 0).any():
            raise ValueError(f"Столбец '{feature}' содержит значения <= 0, логарифмирование невозможно.")
        x = np.log(X[feature]+1)
    else:
        x = X[feature]
    quant_25, quant_75 = x.quantile(0.25), x.quantile(0.75)
    IQR = quant_75 - quant_25
    bond_low = quant_25 - IQR * left
    bond_up = quant_75 + IQR * right

    cleaned_mask = (x >= bond_low) & (x <= bond_up)
    outlier_mask = ~cleaned_mask

    cleaned_data = X[cleaned_mask]
    cleaned_y = y[cleaned_mask]

    outliers_data = X[outlier_mask]
    outliers_y = y[outlier_mask]

    print(f"Удалено {outlier_mask.sum()} объектов из {X.shape[0]}")

    return cleaned_data, cleaned_y, outliers_data
