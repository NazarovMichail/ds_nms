import pandas as pd
from typing import List, Tuple, Any, Dict, Literal, Union
import pickle
import json
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


def save_data(
        file_dict: Dict[str, Any],
        sub_dir: str,
        directory: str = 'data/',
        format: str = 'pkl') -> None:
    """
    Сохраняет данные из словаря в указанную директорию.

    Args:
        file_dict (Dict[str, Any]): Словарь с именами файлов и данными для сохранения.
        sub_dir (str): Поддиректория для сохранения файлов.
        directory (str, optional): Основная директория для сохранения. Defaults to 'data/'.
        format (str, optional): Формат сохранения ('pkl' или 'json'). Defaults to 'pkl'.

    Returns:
        None
    """
    final_dir = os.path.join(directory, sub_dir)
    os.makedirs(final_dir, exist_ok=True)

    for file_name, data in file_dict.items():
        file_path = os.path.join(final_dir, f"{file_name}.{format}")
        try:
            if format == 'pkl':
                with open(file_path, 'wb') as file:
                    pickle.dump(data, file)
            elif format == 'json':
                import json
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
            else:
                raise ValueError(f"Неподдерживаемый формат: {format}")
            print(f"Файл {file_path} успешно сохранён.")
        except Exception as error:
            print(f"Ошибка при сохранении файла {file_name}: {error}")

def load_data(
        file_lst: List[str],
        sub_dir: str,
        directory: str = 'data/',
        format: str = 'pkl') -> List[Any]:
    """
    Загружает данные из указанных файлов.

    Args:
        file_lst (List[str]): Список имён файлов (без расширения).
        sub_dir (str): Поддиректория, где находятся файлы.
        directory (str, optional): Основная директория. Defaults to 'data/'.
        format (str, optional): Формат файлов ('pkl' или 'json'). Defaults to 'pkl'.

    Returns:
        List[Any]: Список загруженных данных.
    """
    loaded_lst = []
    final_dir = os.path.join(directory, sub_dir)

    for file_name in file_lst:
        file_path = os.path.join(final_dir, f"{file_name}.{format}")
        try:
            if format == 'pkl':
                with open(file_path, 'rb') as file:
                    loaded_file = pickle.load(file)
            elif format == 'json':
                with open(file_path, 'r', encoding='utf-8') as file:
                    loaded_file = json.load(file)
            else:
                raise ValueError(f"Неподдерживаемый формат: {format}")

            loaded_lst.append(loaded_file)
            print(f"Файл {file_path} успешно загружен.")

        except FileNotFoundError:
            print(f"Файл {file_path} не найден.")
        except Exception as error:
            print(f"Ошибка при загрузке файла {file_path}: {error}")

    return loaded_lst
