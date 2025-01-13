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
from ds_nms import utils_io


def get_selected_features(
        X: pd.DataFrame,
        y: pd.Series,
        selector: RFE | SequentialFeatureSelector,
        estimator: BaseEstimator,
        n_features_to_select: int = None,
        direction: Literal['forward', 'backward'] = None
    ) -> Tuple[pd.DataFrame, RFE | SequentialFeatureSelector]:
    """
    Выполняет отбор признаков и возвращает новый датафрейм с выбранными признаками.

    Args:
        X (pd.DataFrame): Исходный датафрейм.
        y (pd.Series): Целевая переменная.
        selector (RFE | SequentialFeatureSelector): Алгоритм выбора признаков.
        - RFE - алгоритм, который удаляет признаки итеративно, начиная с полного набора признаков,
        исключая  наименее важные признаки на каждой итерации.
        Подходит только для моделей, которые могут возвращать важность признаков (например, деревья решений, линейные модели).
        - SFS - алгоритм, который итеративно добавляет (или удаляет) признаки к модели один за другим.
        Может работать с любыми моделями, так как основан на метриках оценки качества.

        estimator (BaseEstimator): Модель для оценки. Может учитывать влияние комбинаций признаков,
        так как тестирует их по отдельности.
        n_features_to_select (int, optional): Количество признаков для выбора. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, RFE | SequentialFeatureSelector]: Преобразованные данные и обученный селектор.
    """
    if direction:
        selector_instance = selector(estimator=estimator,
                                    n_features_to_select=n_features_to_select,
                                    direction=direction)
    selector_instance = selector(estimator=estimator, n_features_to_select=n_features_to_select)
    selector_instance.fit(X, y)
    selected_columns = X.columns[selector_instance.support_]
    X_selected = pd.DataFrame(selector_instance.transform(X), columns=selected_columns, index=X.index)
    return X_selected, selector_instance

def get_best_n_features(
        X: pd.DataFrame,
        y: pd.Series,
        selector: RFE | SequentialFeatureSelector,
        scoring: List[str],
        estimator: BaseEstimator,
        X_name: str,
        n_folds: int = 4,
        direction: Literal['forward', 'backward'] = None
    ) -> Tuple[int, optuna.Study]:
    """
    Оптимизирует количество признаков с использованием Optuna.

    Args:
        X (pd.DataFrame): Исходный датафрейм.
        y (pd.Series): Целевая переменная.
        selector (RFE | SequentialFeatureSelector): Алгоритм выбора признаков.
        - RFE - алгоритм, который удаляет признаки итеративно, начиная с полного набора признаков,
        исключая  наименее важные признаки на каждой итерации.
        Подходит только для моделей, которые могут возвращать важность признаков (например, деревья решений, линейные модели).
        - SFS - алгоритм, который итеративно добавляет (или удаляет) признаки к модели один за другим.
        Может работать с любыми моделями, так как основан на метриках оценки качества.

        estimator (BaseEstimator): Модель для оценки. Может учитывать влияние комбинаций признаков,
        так как тестирует их по отдельности.
        scoring (List[str]): cross_validate - метрики для оценки.
        estimator (BaseEstimator): Модель для оценки.
        X_name (str): Имя набора данных.
        sampler (optuna.samplers.BaseSampler, optional): Самплер для Optuna. Defaults to TPESampler().
        n_folds (int): Кол-во фолдов кросс-валидации для подбора оптимального кол-ва признаков
        direction (Literal['forward', 'backward']): Добавление / удаление признаков для SequentialFeatureSelector
    Returns:
        Tuple[int, optuna.Study]: Количество признаков и объект Optuna Study.
    """
    def objective(trial):
        n_features_to_select = trial.suggest_int("n_features_to_select", 1, X.shape[1])
        if direction:
            X_reduced, _ = get_selected_features(X=X,
                                            y=y,
                                            selector=selector,
                                            estimator=estimator,
                                            n_features_to_select=n_features_to_select,
                                            direction=direction)
        X_reduced, _ = get_selected_features(X=X,
                                            y=y,
                                            selector=selector,
                                            estimator=estimator,
                                            n_features_to_select=n_features_to_select)
        result = cross_validate(estimator=estimator,
                                X=X_reduced,
                                y=y,
                                scoring=scoring,
                                cv=KFold(n_folds))
        return -result['test_neg_root_mean_squared_error'].mean()

    grid = {'n_features_to_select': range(1, X.shape[1])}
    study = optuna.create_study(study_name=f"{X_name}_{estimator}_study",
                                direction="minimize",
                                sampler=optuna.samplers.GridSampler(grid))
    study.optimize(objective, n_trials=X.shape[1])
    best_n_features = study.best_params['n_features_to_select']
    return best_n_features, study

def save_selected_features(
        data_dict: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        selector: RFE | SequentialFeatureSelector,
        estimators: List[BaseEstimator],
        scoring: List[str],
        output_dir: str,
        n_folds: int = 4,
        direction: Literal['forward', 'backward'] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
    """
    Выполняет отбор признаков для нескольких наборов данных и нескольких моделей, сохраняет результаты.

    Args:
        data_dict (Dict[str, Tuple[pd.DataFrame, pd.Series]]): Словарь с наборами данных.
        selector (RFE | SequentialFeatureSelector): Алгоритм выбора признаков.
        - RFE - алгоритм, который удаляет признаки итеративно, начиная с полного набора признаков,
        исключая  наименее важные признаки на каждой итерации.
        Подходит только для моделей, которые могут возвращать важность признаков (например, деревья решений, линейные модели).
        - SFS - алгоритм, который итеративно добавляет (или удаляет) признаки к модели один за другим.
        Может работать с любыми моделями, так как основан на метриках оценки качества.

        estimators (List[BaseEstimator]): Список моделей для оценки.
        scoring (List[str]): Метрики для оценки.
        output_dir (str): Директория для сохранения.
        n_folds (int): Кол-во фолдов кросс-валидации для подбора оптимального кол-ва признаков
        direction (Literal['forward', 'backward']): Добавление / удаление признаков для
        verbose (bool, optional): Показывать прогресс. Defaults to True.
    Returns:
        pd.DataFrame: Датафрейм с результатами подбора признаков.
    """
    X_selected_dict = {}
    all_trials = []
    timestamp = dt.now().strftime("%Y_%m_%d_%H_%M")
    output_dir = os.path.join(output_dir, f'f_select_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    for estimator in tqdm(estimators, disable=not verbose):
        for data_name, (X, y) in tqdm(data_dict.items(), disable=not verbose, leave=False):
            best_n_features, study = get_best_n_features(X=X,
                                                    y=y,
                                                    selector=selector,
                                                    scoring=scoring,
                                                    estimator=estimator,
                                                    X_name=data_name,
                                                    n_folds=n_folds,
                                                    direction=direction)
            X_selected, _ = get_selected_features(X=X,
                                                y=y,
                                                selector=selector,
                                                estimator=estimator,
                                                n_features_to_select=best_n_features,
                                                direction=direction)
            X_selected_dict[f"{data_name}_{estimator.__class__.__name__}"] = X_selected

            trials_df = study.trials_dataframe()
            trials_df['data_name'] = data_name
            trials_df['estimator'] = estimator.__class__.__name__
            all_trials.append(trials_df)

    clear_output()
    utils_io.save_data(X_selected_dict, output_dir)

    concatenated_trials_df = pd.concat(all_trials, ignore_index=True)

    return concatenated_trials_df
