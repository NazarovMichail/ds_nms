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








def kolmog_smirn_test(sample_1: pd.Series,
                    sample_2: pd.Series,
                    alpha: float=0.05) -> None:
    """Тест на то, принадлежат ли две выборки одному и тому же распределению

    Args:
        sample_1 (pd.Series): Выборка 1
        sample_2 (pd.Series): Выборка 2
        alpha (float, optional): Вероятность ошибочно отклонить нулевую гипотезу. Defaults to 0.05.
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
        alpha (float, optional): Вероятность ошибочно отклонить нулевую гипотезу. Defaults to 0.05.
    """

    statistic, p_value = kruskal(sample_1, sample_2)
    if p_value < alpha:
        print(f'p-value={p_value:.5f}')
        print("Гипотеза о равенстве распределения отвергается")
    else:
        print(f'p-value={p_value:.5f}')
        print("Данные могут быть взяты из одного распределения")


def df_scaling(df_train: pd.DataFrame,
            df_test: pd.DataFrame,
            numerical_columns: List[str],
            scaler: StandardScaler | MinMaxScaler | Normalizer | RobustScaler,
            return_scaler: bool = False
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:

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

    print(df_train_num_scaled.shape, df_test_num_scaled.shape)
    display(df_train_num_scaled.describe().round(1))
    display(df_test_num_scaled.describe().round(1))

    if return_scaler:
        return df_train_num_scaled, df_test_num_scaled, train_scaler

    return df_train_num_scaled, df_test_num_scaled


def save_data(
            file_dict: Dict[str, Any],
            dir: str,
            store: str='data/',
            ) -> None:


    for file_name, data in file_dict.items():
        try:
            with open(f'{store}{dir}{file_name}.pkl', 'wb') as file:
                pickle.dump(data, file)
                print(f'Файл {file.name} записан')
        except FileNotFoundError as error:
            print(error)
            os.mkdir(f'{store}{dir}')
            with open(f'{store}{dir}{file_name}.pkl', 'wb') as file:
                pickle.dump(data, file)
                print(f'Файл {file.name} записан')


def load_data(
            file_lst: List[str],
            dir: str,
            store: str='data/',
            ) -> List[Any]:

    loaded_lst = []
    for file_name in file_lst:
        try:
            with open(f'{store}{dir}{file_name}.pkl', 'rb') as file:
                loaded_file = pickle.load(file)
                loaded_lst.append(loaded_file)
                print(f'Файл {file.name} загружен')
        except FileNotFoundError as error:
            print(error)
    return loaded_lst


def drop_outliers_iso(X: pd.DataFrame, y: pd.Series,
                contamination: float=0.04,
                n_estimators: int=100) -> Tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame]:

    columns_list = list(X.columns)
    BEFORE_SHAPE = X.shape[0]

    irf = IsolationForest(contamination=contamination,
                    n_estimators=n_estimators,
                    random_state=1)
    irf.fit(X.values)
    prediction = irf.predict(X.values)

    clear_inx = np.where(prediction == 1)
    outlier_inx = np.where(prediction == -1)

    X_cleared = X.to_numpy()[clear_inx]
    y_cleared = y.to_numpy()[clear_inx]
    outliers = X.to_numpy()[outlier_inx]

    AFTER_SHAPE = X_cleared.shape[0]

    print(f"Удалено {BEFORE_SHAPE - AFTER_SHAPE} объектов")

    return pd.DataFrame(X_cleared, columns=columns_list).set_index(list(clear_inx)),\
        pd.DataFrame(y_cleared).set_index(list(clear_inx)),\
    pd.DataFrame(outliers, columns=columns_list, index=list(outlier_inx))


def drop_outliers_tuk(data: pd.DataFrame,
                feature : str,
                left:float=1.5, right:float=1.5,
                log_scale:bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Очищает датафрейм от выбросов по методу Тьюки

    Args:
        data (pd.DataFrame): датафрейм
        feature (str): столбец
        left (float, optional): число влево от IQR. Defaults to 1.5.
        right (float, optional): число вправо от IQR. Defaults to 1.5.

    Returns:
        tuple: датафрейм с выбросами,
        очищенный датафрейм от выбросов
    """
    if log_scale:
        x = np.log(data[feature]+1)
    else:
        x = data[feature]
    quant_25, quant_75 = x.quantile(0.25), x.quantile(0.75)
    IQR = quant_75 - quant_25
    bond_low = quant_25 - IQR * left
    bond_up = quant_75 + IQR * right
    outliers = data[ (x < bond_low )| (x > bond_up )]
    cleaned_data = data[(x >= bond_low) & (x <= bond_up)]
    return cleaned_data, outliers


def get_selected_features(X: pd.DataFrame,
            y: pd.Series,
            selector: RFE | SequentialFeatureSelector,
            estimator: BaseEstimator,
            n_features_to_select=None
            ) -> Tuple[pd.DataFrame, RFE | SequentialFeatureSelector]:

    selector_init = selector(estimator=estimator, n_features_to_select=n_features_to_select)
    selector_trained = selector_init.fit(X, y)
    selected_inx = selector_trained.support_
    selected_columns = list(X.columns[selected_inx])
    result_df = pd.DataFrame(selector_trained.transform(X), columns=selected_columns)

    return result_df, selector_trained


def get_best_n_features(
                        X: pd.DataFrame,
                        y: pd.Series,
                        selector: RFE | SequentialFeatureSelector,
                        scoring: dict,
                        estimator: BaseEstimator,
                        X_name: str,
                        ) -> Tuple[int, optuna.Study]:

    samples_num = X.shape[1]
    def objective(trial):

        n_features_to_select = trial.suggest_int("n_features_to_select", 1, samples_num)

        X_reduced, _ = get_selected_features(X, y,
                    selector,
                    estimator,
                    n_features_to_select=n_features_to_select)

        result = cross_validate(estimator,
                            X_reduced,
                            y, scoring=list(scoring.values()),
                            cv=KFold(4),
                            return_estimator=True
                            )

        return -result['test_neg_root_mean_squared_error'].mean()

    grid = {'n_features_to_select': range(1, samples_num)}
    study = optuna.create_study(study_name=f"{estimator}_{X_name}_study",
                            direction="minimize",
                            sampler=optuna.samplers.GridSampler(grid),
                            )
    study.optimize(objective, n_trials=samples_num)
    trial = study.best_trial
    params = trial.params
    selected_n_features = params['n_features_to_select']

    return selected_n_features, study


def save_selected_features(data_dict: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                            selector: RFE | SequentialFeatureSelector,
                            estimators_lst: List[BaseEstimator],
                            scoring: Dict[str, str],
                            dir: str
                            ):

    X_selected_dict = {}
    X_selected_names_lst = []

    progress_bar = tqdm(estimators_lst)
    for estimator in progress_bar:
        data_progress_bar = tqdm(data_dict.items())
        for data_name, data in data_progress_bar:
            X, y = data
            selected_n_features, _ = get_best_n_features(X, y, selector, scoring, estimator, data_name)
            X_selected, selector_trained  = get_selected_features(X, y, selector, estimator, selected_n_features)

            X_name = f'{data_name}_{estimator}'

            X_selected_dict[X_name] = X_selected
            X_selected_names_lst.append(X_name)

            clear_output()

    X_names_dict = {'Selected_features_names': X_selected_names_lst }
    dir = f'{dir}{dt.now().strftime("%Y_%m_%d_%H_%M")}/'

    clear_output()
    save_data(X_selected_dict, dir)
    save_data(X_names_dict, dir)


def train_stratify_cv(
                    X: pd.DataFrame,
                    y: pd.Series,
                    model: BaseEstimator,
                    stratify: str='quantile',
                    get_coefs: bool=True,
                    data_name: str=None
                    ) -> Tuple[BaseEstimator, dict] | Tuple[BaseEstimator, dict, pd.DataFrame]:

    QUANT_25 = y.quantile(0.25)
    QUANT_50 = y.quantile(0.50)
    QUANT_75 = y.quantile(0.75)

    if stratify == 'quantile':
        strat = y.apply(lambda y: 1 if y < QUANT_25 else 2 if QUANT_25 < y < QUANT_50 else 3 if QUANT_50 < y < QUANT_75 else 4  )
    else:
        strat = X[stratify]

    R2_test_results = []
    RMSE_test_results = []
    MAE_test_results = []
    R2_train_results = []
    RMSE_train_results = []
    MAE_train_results = []

    models_result = []
    result= {}

    skf = StratifiedKFold(n_splits=4,
                        shuffle=True,
                        random_state=1)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    for train_index, test_index in skf.split(X, strat):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        R2_train = r2_score(y_train, y_train_pred)
        RMSE_train = mean_squared_error(y_train, y_train_pred)**0.5
        MAE_train = mean_absolute_error(y_train, y_train_pred)

        R2_test = r2_score(y_test, y_test_pred)
        RMSE_test = mean_squared_error(y_test, y_test_pred)**0.5
        MAE_test = mean_absolute_error(y_test, y_test_pred)

        R2_train_results.append(R2_train)
        RMSE_train_results.append(RMSE_train)
        MAE_train_results.append(MAE_train)

        R2_test_results.append(R2_test)
        RMSE_test_results.append(RMSE_test)
        MAE_test_results.append(MAE_test)

        models_result.append(model)

    result['R2_train'] = np.array(R2_train_results)
    result['RMSE_train'] = np.array(RMSE_train_results)
    result['MAE_train'] = np.array(MAE_train_results)

    result['R2_test'] = np.array(R2_test_results)
    result['RMSE_test'] = np.array(RMSE_test_results)
    result['MAE_test'] = np.array(MAE_test_results)

    final_result = {}
    best_score_ind = result['R2_test'].argmax()
    best_model = models_result[best_score_ind]

    final_result['model'] = f"{model}"
    for name, metric in result.items():
        final_result[f'{name}_mean'] = result[name].mean().round(2)
        final_result[f'{name}_std'] = result[name].std().round(2)
        final_result[f'{name}_splits'] = result[name].round(2)

    final_result['data_name'] = data_name
    if get_coefs:
        try:
            features_sorted_ind = np.abs((best_model.coef_)).argsort()[::-1]
            important_columns = list(X.columns[features_sorted_ind])
            importance_coefs = best_model.coef_.round(1)[features_sorted_ind]
            feature_importances_df = pd.DataFrame(importance_coefs, index=important_columns, columns=['importances']).reset_index().rename(columns={'index': 'features'})
        except AttributeError as error:
            df_columns = X.columns
            importances = best_model.feature_importances_
            importances_df = pd.DataFrame(importances, index=df_columns, columns=['importances'])
            feature_importances_df = importances_df.sort_values(by='importances', ascending=False).reset_index().rename(columns={'index': 'features'})

        return best_model, final_result, feature_importances_df

    return best_model, final_result


def train_loo_cv(
                    X: pd.DataFrame,
                    y: pd.Series,
                    model: BaseEstimator,
                    get_coefs: bool=False,
                    data_name: str=None
                    ) -> Tuple[BaseEstimator, dict] | Tuple[BaseEstimator, dict, pd.DataFrame]:

    R2_train_results = []
    y_val_results = []
    RMSE_val_results = []
    NRMSE_val_results = []
    MAE_val_results = []
    RE_val_results = []
    y_pred_results = []
    RMSE_train_results = []
    NRMSE_train_results = []
    MAE_train_results = []
    RE_train_results = []

    models_result = []
    result = {}

    loo = LeaveOneOut()

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    progrbar_splits = tqdm(loo.split(X), total=len(X))
    for train_index, val_index in progrbar_splits:
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)

        result_train, y_train_pred = get_prediction(X_train, y_train, model)
        result_val, y_val_pred = get_prediction(X_val, y_val, model)

        y_val_results.append(y_val)
        y_pred_results.append(y_val_pred)

        R2_train = result_train['R2']
        RMSE_train = result_train['RMSE']
        NRMSE_train = result_train['NRMSE']
        MAE_train = result_train['MAE']
        RE_train = result_train['RE']

        R2_train_results.append(R2_train)
        RMSE_train_results.append(RMSE_train)
        NRMSE_train_results.append(NRMSE_train)
        MAE_train_results.append(MAE_train)
        RE_train_results.append(RE_train)

        RMSE_val = result_val['RMSE']
        MAE_val = result_val['MAE']
        RE_val = result_val['RE']

        RMSE_val_results.append(RMSE_val)
        MAE_val_results.append(MAE_val)
        RE_val_results.append(RE_val)

        models_result.append(model)

    result['R2_train'] = np.array(R2_train_results)
    result['RMSE_train'] = np.array(RMSE_train_results)
    result['NRMSE_train'] = np.array(NRMSE_train_results)
    result['MAE_train'] = np.array(MAE_train_results)
    result['RE_train'] = np.array(RE_train_results)

    result['MAE_val'] = np.array(MAE_val_results)
    result['RE_val'] = np.array(RE_val_results)

    final_result = {}
    best_score_ind = result['MAE_val'].argmin()
    best_model = models_result[best_score_ind]

    final_result['model'] = f"{model}"
    for name, metric in result.items():
        final_result[f'{name}_mean'] = result[name].mean().round(2)
        final_result[f'{name}_std'] = result[name].std().round(2)
        final_result[f'{name}_splits'] = result[name].round(2)

    final_result['R2_val_mean'] = r2_score(np.array(y_val_results), np.array(y_pred_results))
    final_result['RMSE_val_mean'] = mean_squared_error(np.array(y_val_results), np.array(y_pred_results))**0.5
    final_result['NRMSE_val_mean'] = final_result['RMSE_val_mean'] / (np.array(y_val_results).max() - np.array(y_val_results).min())

    final_result['data_name'] = data_name
    if get_coefs:
        try:
            features_sorted_ind = np.abs((best_model.coef_)).argsort()[::-1]
            important_columns = list(X.columns[features_sorted_ind])
            importance_coefs = best_model.coef_.round(1)[features_sorted_ind]
            feature_importances_df = pd.DataFrame(importance_coefs, index=important_columns, columns=['importances']).reset_index().rename(columns={'index': 'features'})
        except AttributeError as error:
            df_columns = X.columns
            importances = best_model.feature_importances_
            importances_df = pd.DataFrame(importances, index=df_columns, columns=['importances'])
            feature_importances_df = importances_df.sort_values(by='importances', ascending=False).reset_index().rename(columns={'index': 'features'})

        clear_output()
        return best_model, final_result, feature_importances_df

    clear_output()
    return best_model, final_result


def train_KF_cv(
                    X: pd.DataFrame,
                    y: pd.Series,
                    model: BaseEstimator,
                    get_coefs: bool=False,
                    data_name: str=None
                    ) -> Tuple[BaseEstimator, dict] | Tuple[BaseEstimator, dict, pd.DataFrame]:

    R2_train_results = []
    R2_val_results = []
    y_val_results = []
    RMSE_val_results = []
    NRMSE_val_results = []
    MAE_val_results = []
    RE_val_results = []
    y_pred_results = []
    RMSE_train_results = []
    NRMSE_train_results = []
    MAE_train_results = []
    RE_train_results = []

    models_result = []
    result = {}

    KFCV = KFold(n_splits=5, shuffle=False)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    progrbar_splits = tqdm(KFCV.split(X), total=len(X))
    for train_index, val_index in progrbar_splits:
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)

        result_train, y_train_pred = get_prediction(X_train, y_train, model)
        result_val, y_val_pred = get_prediction(X_val, y_val, model)

        y_val_results.append(y_val.values)
        y_pred_results.append(y_val_pred)

        R2_train = result_train['R2']
        RMSE_train = result_train['RMSE']
        NRMSE_train = result_train['NRMSE']
        MAE_train = result_train['MAE']
        RE_train = result_train['RE']

        R2_train_results.append(R2_train)
        RMSE_train_results.append(RMSE_train)
        NRMSE_train_results.append(NRMSE_train)
        MAE_train_results.append(MAE_train)
        RE_train_results.append(RE_train)

        R2_val = result_val['R2']
        RMSE_val = result_val['RMSE']
        NRMSE_val = result_val['NRMSE']
        MAE_val = result_val['MAE']
        RE_val = result_val['RE']

        R2_val_results.append(R2_val)
        RMSE_val_results.append(RMSE_val)
        NRMSE_val_results.append(NRMSE_val)
        MAE_val_results.append(MAE_val)
        RE_val_results.append(RE_val)

        models_result.append(model)

    result['R2_train'] = np.array(R2_train_results)
    result['RMSE_train'] = np.array(RMSE_train_results)
    result['NRMSE_train'] = np.array(NRMSE_train_results)
    result['MAE_train'] = np.array(MAE_train_results)
    result['RE_train'] = np.array(RE_train_results)

    result['R2_val'] = np.array(R2_val_results)
    result['RMSE_val'] = np.array(RMSE_val_results)
    result['NRMSE_val'] = np.array(NRMSE_val_results)
    result['MAE_val'] = np.array(MAE_val_results)
    result['RE_val'] = np.array(RE_val_results)

    final_result = {}
    best_score_ind = result['MAE_val'].argmin()
    best_model = models_result[best_score_ind]

    final_result['model'] = f"{model}"
    for name, metric in result.items():
        final_result[f'{name}_mean'] = result[name].mean().round(2)
        final_result[f'{name}_std'] = result[name].std().round(2)
        final_result[f'{name}_splits'] = result[name].round(2)

    final_result['data_name'] = data_name
    if get_coefs:
        try:
            features_sorted_ind = np.abs((best_model.coef_)).argsort()[::-1]
            important_columns = list(X.columns[features_sorted_ind])
            importance_coefs = best_model.coef_.round(1)[features_sorted_ind]
            feature_importances_df = pd.DataFrame(importance_coefs, index=important_columns, columns=['importances']).reset_index().rename(columns={'index': 'features'})
        except AttributeError as error:
            df_columns = X.columns
            importances = best_model.feature_importances_
            importances_df = pd.DataFrame(importances, index=df_columns, columns=['importances'])
            feature_importances_df = importances_df.sort_values(by='importances', ascending=False).reset_index().rename(columns={'index': 'features'})

        clear_output()
        return best_model, final_result, feature_importances_df

    clear_output()
    return best_model, final_result



def train_TS_cv(
                    X: pd.DataFrame,
                    y: pd.Series,
                    model: BaseEstimator,
                    get_coefs: bool=False,
                    data_name: str=None,
                    train_size: int=48,
                    val_size: int=12,
                    ) -> Tuple[BaseEstimator, dict] | Tuple[BaseEstimator, dict, pd.DataFrame]:

    R2_train_results = []
    R2_val_results = []
    y_val_results = []
    RMSE_val_results = []
    NRMSE_val_results = []
    MAE_val_results = []
    RE_val_results = []
    y_pred_results = []
    RMSE_train_results = []
    NRMSE_train_results = []
    MAE_train_results = []
    RE_train_results = []

    models_result = []
    result = {}

    n_splits = (len(X) - train_size) // val_size
    tscv = TimeSeriesSplit(n_splits=n_splits,test_size=val_size)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    progrbar_splits = tqdm(tscv.split(X), total=len(X))
    for train_index, val_index in progrbar_splits:
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)

        result_train, y_train_pred = get_prediction(X_train, y_train, model)
        result_val, y_val_pred = get_prediction(X_val, y_val, model)

        y_val_results.append(y_val.values)
        y_pred_results.append(y_val_pred)

        R2_train = result_train['R2']
        RMSE_train = result_train['RMSE']
        NRMSE_train = result_train['NRMSE']
        MAE_train = result_train['MAE']
        RE_train = result_train['RE']

        R2_train_results.append(R2_train)
        RMSE_train_results.append(RMSE_train)
        NRMSE_train_results.append(NRMSE_train)
        MAE_train_results.append(MAE_train)
        RE_train_results.append(RE_train)

        R2_val = result_val['R2']
        RMSE_val = result_val['RMSE']
        NRMSE_val = result_val['NRMSE']
        MAE_val = result_val['MAE']
        RE_val = result_val['RE']

        R2_val_results.append(R2_val)
        RMSE_val_results.append(RMSE_val)
        NRMSE_val_results.append(NRMSE_val)
        MAE_val_results.append(MAE_val)
        RE_val_results.append(RE_val)

        models_result.append(model)

    result['R2_train'] = np.array(R2_train_results)
    result['RMSE_train'] = np.array(RMSE_train_results)
    result['NRMSE_train'] = np.array(NRMSE_train_results)
    result['MAE_train'] = np.array(MAE_train_results)
    result['RE_train'] = np.array(RE_train_results)

    result['R2_val'] = np.array(R2_val_results)
    result['RMSE_val'] = np.array(RMSE_val_results)
    result['NRMSE_val'] = np.array(NRMSE_val_results)
    result['MAE_val'] = np.array(MAE_val_results)
    result['RE_val'] = np.array(RE_val_results)

    final_result = {}
    best_model = models_result[-1]

    final_result['model'] = f"{model}"
    for name, metric in result.items():
        final_result[f'{name}_mean'] = result[name].mean().round(2)
        final_result[f'{name}_std'] = result[name].std().round(2)
        final_result[f'{name}_splits'] = result[name].round(2)

    final_result['data_name'] = data_name
    if get_coefs:
        try:
            features_sorted_ind = np.abs((best_model.coef_)).argsort()[::-1]
            important_columns = list(X.columns[features_sorted_ind])
            importance_coefs = best_model.coef_.round(1)[features_sorted_ind]
            feature_importances_df = pd.DataFrame(importance_coefs, index=important_columns, columns=['importances']).reset_index().rename(columns={'index': 'features'})
        except AttributeError as error:
            df_columns = X.columns
            importances = best_model.feature_importances_
            importances_df = pd.DataFrame(importances, index=df_columns, columns=['importances'])
            feature_importances_df = importances_df.sort_values(by='importances', ascending=False).reset_index().rename(columns={'index': 'features'})

        clear_output()
        return best_model, final_result, feature_importances_df

    clear_output()
    return best_model, final_result


def train_single_cv(
                    X: pd.DataFrame,
                    y: pd.Series,
                    model: BaseEstimator,
                    scoring: Dict[str, str],
                    get_coefs: bool=True,
                    data_name: str=None
                    ) -> Tuple[BaseEstimator, dict] | Tuple[BaseEstimator, dict, pd.DataFrame]:

    result = cross_validate(model,
                            X, y,
                            scoring=list(scoring.values()),
                            cv=KFold(5, shuffle=True),
                            return_estimator=True,
                            )

    result_dict = {}
    best_score_ind = result['test_neg_root_mean_squared_error'].argmax()
    best_model = result['estimator'][best_score_ind]

    result_dict['model'] = f"{model}"
    for name, metric in scoring.items():
        if name != 'R2':
            result_dict[f'{name}_mean'] = -result[f'test_{metric}'].mean().round(2)
            result_dict[f'{name}_std'] = result[f'test_{metric}'].std().round(2)
            result_dict[f'{name}_splits'] = result[f'test_{metric}'].round(2)*-1
        else:
            result_dict[f'{name}_mean'] = result[f'test_{metric}'].mean().round(2)
            result_dict[f'{name}_std'] = result[f'test_{metric}'].std().round(2)
            result_dict[f'{name}_splits'] = result[f'test_{metric}'].round(2)

    result_dict['data_name'] = data_name
    if get_coefs:
        try:
            features_sorted_ind = np.abs((best_model.coef_)).argsort()[::-1]
            important_columns = list(X.columns[features_sorted_ind])
            importance_coefs = best_model.coef_.round(1)[features_sorted_ind]
            feature_importances_df = pd.DataFrame(importance_coefs, index=important_columns, columns=['importances']).reset_index().rename(columns={'index': 'features'})
        except AttributeError as error:
            df_columns = X.columns
            importances = best_model.feature_importances_
            importances_df = pd.DataFrame(importances, index=df_columns, columns=['importances'])
            feature_importances_df = importances_df.sort_values(by='importances', ascending=False).reset_index().rename(columns={'index': 'features'})

        return best_model, result_dict, feature_importances_df

    return best_model, result_dict


def train_several_cv(X: pd.DataFrame,
                        y: pd.Series,
                        models_list: List[BaseEstimator],
                        scoring: Dict[str, str],
                        data_name: str=None) -> pd.DataFrame:

    results_list  = []
    for model in models_list:

        print(f"-------------> {model}")

        best_model, result_dict = train_loo_cv(
                                                    X, y,
                                                    model,
                                                    get_coefs=False,
                                                    data_name=data_name)
        results_list.append(result_dict)

    clear_output()
    result_df = pd.DataFrame(results_list)
    return result_df


def train_several_cv_list(X_lst: List[pd.DataFrame],
                        target: pd.Series,
                        X_name_lst: List[str],
                        models_list: List[BaseEstimator],
                        scoring: Dict[str, str],
                        ) -> pd.DataFrame:

    X_count = len(X_lst)

    result_df_lst = []
    for i in range(X_count):

        X = X_lst[i]
        X_name = X_name_lst[i]

        result_df = train_several_cv(X, target,
                                    models_list,
                                    scoring)

        result_df['data_name'] = X_name
        result_df_lst.append(result_df)

    all_results_df = pd.concat(result_df_lst)
    return all_results_df


def get_importances_barplot(X: pd.DataFrame,
                            y: pd.Series,
                            model: BaseEstimator,
                            model_name: str="model",
                            for_best_split: bool=True,
                            ) -> pd.DataFrame:

    if for_best_split:
        best_model, metrics_dict, feature_importances_df =  train_loo_cv(X, y, model, get_coefs=True)
        feature_importances_df['importances_rel'] = round(feature_importances_df['importances'] / abs(feature_importances_df['importances']).sum(), 2)

    else:
        model.fit(X, y)
        try:
            features_sorted_ind = np.abs((model.coef_)).argsort()[::-1]
            important_columns = list(X.columns[features_sorted_ind])
            importance_coefs = model.coef_.round(1)[features_sorted_ind]
            feature_importances_df = pd.DataFrame(importance_coefs, index=important_columns, columns=['importances']).reset_index().rename(columns={'index': 'features'})
            feature_importances_df['importances_rel'] = round(feature_importances_df['importances'] / abs(feature_importances_df['importances']).sum(), 2)
        except AttributeError as error:
            df_columns = X.columns
            importances = model.feature_importances_
            importances_df = pd.DataFrame(importances, index=df_columns, columns=['importances'])
            feature_importances_df = importances_df.sort_values(by='importances', ascending=False).reset_index().rename(columns={'index': 'features'})
            feature_importances_df['importances_rel'] = round(feature_importances_df['importances'] / abs(feature_importances_df['importances']).sum(), 2)

    figure, ax = plt.subplots(1,1, figsize=(20, 10))
    ax = sns.barplot(data=feature_importances_df, y='features', x='importances_rel', ax=ax, orient='h', color='r', edgecolor='black', width=0.3)
    for p in ax.patches:
        ax.annotate(format(p.get_width(), '.2f'),  # Значение на конце столбца
                (p.get_width(), p.get_y() + p.get_height() / 2),  # Позиция текста
                ha='left', va='center',
                xytext=(2, 0),  # Смещение текста вправо
                textcoords='offset points')
    if for_best_split:
        ax.set_title(f'Feature importances for best split. {model_name}')
    else:
        ax.set_title(f'Feature importances for all dataset. {model_name}')
    ax.grid(axis='x')
    plt.savefig(f"FE_bar_coefs_{model_name}.png", dpi=300)
    plt.show()
    return feature_importances_df


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


def optuna_plot(study: Any, dir: str, param_importances: bool=False, first_param: str="first_param", second_param: str="second_param", model_name="model_name") -> None:

    DATE = dt.now().strftime("%Y_%m_%d_%H_%M")

    if not os.path.exists("optuna_plots"):
        os.mkdir("optuna_plots")
    try:
        optuna.visualization.plot_pareto_front(study, target_names=[f"{first_param}", f"{second_param}"]).write_html(f'optuna_plots/{dir}{model_name}_pareto_front_{DATE}.html')
        optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[0], target_name=f'{first_param}').write_html(f'optuna_plots/{dir}{model_name}_diff_plot_slice_{DATE}.html')
        optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[1], target_name=f'{second_param}').write_html(f'optuna_plots/{dir}{model_name}_metrics_plot_slice_{DATE}.html')

        if param_importances:
            optuna.visualization.plot_param_importances(study).write_html(f'optuna_plots/{dir}param_importances_{DATE}.html')

    except FileNotFoundError:
        dir_path = os.path.join("optuna_plots", f"{dir}")
        os.mkdir(dir_path)
        print("________________________________________________________________________________________________")
        print(f"Создана директория: {dir} для размещения графиков оптимизации гиперпараметров")
        optuna.visualization.plot_pareto_front(study, target_names=[f"{first_param}", f"{second_param}"]).write_html(f'optuna_plots/{dir}{model_name}_pareto_front_{DATE}.html')
        optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[0], target_name=f'{first_param}').write_html(f'optuna_plots/{dir}{model_name}_diff_plot_slice_{DATE}.html')
        optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[1], target_name=f'{second_param}').write_html(f'optuna_plots/{dir}{model_name}_metrics_plot_slice_{DATE}.html')

        if param_importances:
            optuna.visualization.plot_param_importances(study).write_html(f'optuna_plots/{dir}{model_name}_param_importances_{DATE}.html')


def optimize_PAR(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"], n_trials: int=100) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize
    def objective(trial):
        # предложение гиперпараметров
        n_iter_no_change =trial.suggest_int("n_iter_no_change", 1, 5)
        validation_fraction = trial.suggest_float("validation_fraction", 0., 1, log=False)
        C = trial.suggest_float("C", 0.1, 5, log=False)
        epsilon = trial.suggest_float("epsilon", 0, 5, log=False)

        model = PassiveAggressiveRegressor(
                    C=C,
                    epsilon=epsilon,
                    validation_fraction=validation_fraction,
                    n_iter_no_change=n_iter_no_change,
                    max_iter=105000,
                    random_state=1)

        mdl, res = train_loo_cv(X_train, y_train, model, get_coefs=False)

        R2_test_mean = res['R2_val']
        r2_train_mean = res['R2_train_mean']

        RMSE_test_mean = res['RMSE_val']
        RMSE_train_mean = res['RMSE_train_mean']

        MAE_test_mean = res['MAE_val_mean']
        MAE_train_mean = res['MAE_train_mean']

        r2_diff = abs(r2_train_mean - R2_test_mean)
        r2_relative_diff = r2_diff / r2_train_mean
        RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
        MAE_diff = abs(MAE_train_mean - MAE_test_mean)

        y_pred = mdl.predict(X_ALL)
        negative_count = (y_pred < 0).sum()

        if minimize == "negative_count":

            return negative_count, R2_test_mean

        if minimize == "r2_diff":

            return r2_relative_diff, R2_test_mean

        if minimize == "RMSE_diff":

            return RMSE_diff, R2_test_mean

        if minimize == "MAE_diff":

            return MAE_diff, R2_test_mean

    study = optuna.create_study(study_name="params_study",
                                directions=[ "minimize", "maximize"],
                                sampler=optuna.samplers.TPESampler(),
                                # pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials)
    trial = study.best_trials



    results_df = study.trials_dataframe()
    min_minimize_param_df = results_df[results_df['values_0'] == results_df['values_0'].min() ]
    min_minimize_param = min_minimize_param_df['values_0'].min()
    best_results = min_minimize_param_df[min_minimize_param_df['values_1'] == min_minimize_param_df['values_1'].max()]
    best_ind = best_results.index[0]
    params = study.get_trials()[best_ind].params

    clear_output()
    print("Parameters updated !")
    print(f"Minimizing parameters ------> {min_minimize_param}")
    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean")

    return params, min_minimize_param, study


def feature_drop_PAR(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, params: dict, important_columns: List[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame, float]:

    columns_lst = list(X_train.columns)
    params = params

    def objective(trial):

        column = trial.suggest_categorical('column', columns_lst)

        columns_drop = [column]

        X_train_dropped = X_train.drop(columns_drop,axis=1)
        X_test_dropped = X_test.drop(columns_drop,axis=1)
        X_ALL_dropped = pd.concat([X_train_dropped, X_test_dropped])

        model = PassiveAggressiveRegressor(
                    **params,
                    max_iter=105000,
                    random_state=1)

        mdl, res = train_loo_cv(X_train_dropped, y_train, model, get_coefs=False)

        R2_test_mean = res['R2_val']
        r2_train_mean = res['R2_train_mean']

        RMSE_test_mean = res['RMSE_val']
        RMSE_train_mean = res['RMSE_train_mean']

        MAE_test_mean = res['MAE_val_mean']
        MAE_train_mean = res['MAE_train_mean']

        r2_diff = abs(r2_train_mean - R2_test_mean)
        RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
        MAE_diff = abs(MAE_train_mean - MAE_test_mean)

        r2_relative_diff = r2_diff / r2_train_mean

        y_pred = mdl.predict(X_ALL_dropped)
        negative_count = (y_pred < 0).sum()


        return negative_count, R2_test_mean


    grid = {'column': columns_lst}
    study = optuna.create_study(study_name="PAR_study",
                                directions=[ "minimize", "maximize"],
                                sampler=optuna.samplers.GridSampler(grid)
                                )
    study.optimize(objective, n_trials=len(X_train))
    trial = study.best_trials

    results_df = study.trials_dataframe()
    if important_columns != None:
        for imp_col in important_columns:
            results_imp = results_df[results_df['params_column'] != imp_col]
            results_df = results_imp

    min_negative_count_df = results_df[results_df['values_0'] == results_df['values_0'].min() ]
    min_negative_count = min_negative_count_df['values_0'].min()

    best_results = min_negative_count_df[min_negative_count_df['values_1'] == min_negative_count_df['values_1'].max()]

    try:
        best_ind = best_results.index[0]
        params = study.get_trials()[best_ind].params
        column_to_drop = params['column']
    except IndexError as error:
        print(error)
        print("Все столбцы были удалены")
        column_to_drop = []

    X_train_dropped = X_train.drop(column_to_drop, axis=1)
    X_test_dropped = X_test.drop(column_to_drop,axis=1)

    clear_output()
    print("Columns updated !")
    print(f"Minimizing parameters ------> {min_negative_count}")
    optuna_plot(study, dir="columns_drop_plots/", param_importances=False, first_param="negative_count", second_param="R2_test_mean")

    return X_train_dropped, X_test_dropped, min_negative_count


def delete_negatives_PAR(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, important_columns: List[str]=None, n_trials: int=100) -> Tuple[pd.DataFrame, pd.DataFrame]:

    neg_count = 1
    loops_count = 0
    while neg_count != 0:
        loops_count += 1
        columns_count = X_train.shape[1]

        getted_params, neg_count, study = optimize_PAR(X_train, y_train, X_test, "negative_count", n_trials=n_trials)
        if neg_count == 0:
            return X_train, X_test

        X_train, X_test, neg_count = feature_drop_PAR(X_train, y_train, X_test, getted_params, important_columns=important_columns)
        if neg_count == 0:
            break
        if columns_count == X_train.shape[1]:
            print("The columns contain important features to continue feature selection!")
            break

    print(f"Total loops {loops_count}")

    return X_train, X_test


def predicted_to_df(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame | None:

    y_true_index = y_true.index
    y_pred_df = pd.DataFrame(y_pred, index=y_true_index)

    check_index = (y_true.index == y_pred_df.index).all()
    if check_index:
        print(f"Индексы y_pred и y_true совпадают: {check_index}")
        return y_pred_df

    else:
        print(f"Индексы y_pred и y_true не совпадают !: {check_index}")



def get_prediction(X_test: pd.DataFrame, y_test: pd.Series, model: BaseEstimator) -> Tuple[pd.DataFrame, np.ndarray]:

    metrics_dict = {}

    y_pred = model.predict(X_test)

    RMSE = mean_squared_error(y_test, y_pred)**0.5
    MAE = mean_absolute_error(y_test, y_pred)
    negative = (y_pred < 0).sum()

    if y_pred.shape[0] > 1:
        R2 = r2_score(y_test, y_pred)
        NRMSE = RMSE / (y_test.max() - y_test.min()).round(2)

    re = (y_pred - y_test).abs()
    re_relative = (re / y_pred) * 100
    count_less_30 = re_relative[ re_relative < 30].shape[0]
    re_less_30 = count_less_30 / y_test.shape[0]

    if y_pred.shape[0] > 1:
        metrics_dict["R2"] = R2
        metrics_dict["NRMSE"] = NRMSE.round(2)
    metrics_dict["RMSE"] = RMSE.round(2)
    metrics_dict["MAE"] = MAE.round(2)
    metrics_dict["RE"] = re_less_30
    metrics_dict["negative"] = negative

    return pd.DataFrame( metrics_dict, index=[0]), y_pred


def predicted_to_df(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame | None:

    y_true_index = y_true.index
    y_pred_df = pd.DataFrame(y_pred, index=y_true_index)

    check_index = (y_true.index == y_pred_df.index).all()
    if check_index:
        print(f"Индексы y_pred и y_true совпадают: {check_index}")
        return y_pred_df

    else:
        print(f"Индексы y_pred и y_true не совпадают !: {check_index}")


def make_description_df( y_true: pd.Series, y_pred: pd.DataFrame,
                        file_dir: str,
                        map_dict: Dict[str, Tuple[int, str]], rename_columns: Dict[str, str],
                        passflow_type: str,
                        get_test_selection: pd.DataFrame=None,
                        set_test_column: pd.DataFrame=None) -> pd.DataFrame:

    try:
        base_df = pd.read_pickle(f"{file_dir}.pkl")
    except FileNotFoundError:
        base_df = pd.read_excel(f'{file_dir}.xlsx')

    for col_name, change_items in map_dict.items():
        base_df[change_items[0]] = base_df[col_name].map(change_items[1])

    for col_name, rename_col in rename_columns.items():
        base_df[rename_col] = base_df[col_name]

    base_df[f'{passflow_type} пассажиропоток (действительный)'] = y_true.round(0)
    base_df[f'{passflow_type} пассажиропоток (прогноз)'] = y_pred

    base_df['Отклонение (абсолютное)'] = (base_df[f'{passflow_type} пассажиропоток (прогноз)'] - base_df[f'{passflow_type} пассажиропоток (действительный)'])
    base_df['Отклонение (относительное), %'] = (base_df['Отклонение (абсолютное)'] /  base_df[f'{passflow_type} пассажиропоток (действительный)'] * 100).round(1)


    base_df_sorted = base_df.sort_values(by='Отклонение (относительное), %', ascending=False, key=abs)
    base_df_sorted_filtered = base_df_sorted.iloc[:, -6:]

    if get_test_selection is not None:
        base_df_sorted_filtered = base_df_sorted_filtered.loc[get_test_selection.index, :].sort_values(by='Отклонение (относительное), %', ascending=False, key=abs)

    if set_test_column is not None:
        base_df_sorted_filtered['Тест'] = " "

        base_df_sorted_filtered.loc[set_test_column.index, 'Тест'] = "X"

    print(f"Наименьшее значение отклонения: {base_df_sorted_filtered['Отклонение (относительное), %'].min()}")
    print(f"Наибольшее значение отклонения: {base_df_sorted_filtered['Отклонение (относительное), %'].max()}")

    total_samples = base_df_sorted_filtered.shape[0]
    less_30_RE = base_df_sorted_filtered[base_df_sorted_filtered['Отклонение (относительное), %'].abs() < 30].shape[0]
    relative_count_less_30_RE = less_30_RE / total_samples

    print(f"Доля значений с относительными отклонениями меньше 30 %: {relative_count_less_30_RE:.2f}")

    return base_df_sorted_filtered


def get_best_study_params(study: optuna.study.study.Study, threshold: float | int, direction: Literal["minimize", "maximize"]="maximize"):

    results_df = study.trials_dataframe()
    diff_treshold_df = results_df[results_df['values_0'] < threshold]
    if direction == "maximize":
        best_results = diff_treshold_df[diff_treshold_df['values_1'] == diff_treshold_df['values_1'].max()]
    else:
        best_results = diff_treshold_df[diff_treshold_df['values_1'] == diff_treshold_df['values_1'].min()]
    try:
        best_ind = best_results.index[0]
        params = study.get_trials()[best_ind].params
    except IndexError as error:
        print("_"*87)
        print(error)
        print("The threshold value_0 is higher than the set one!. The most minimal value_0 is shown !")
        print("_"*87)

        diff_treshold_df = results_df[results_df['values_0'] == results_df['values_0'].min() ]
        if direction == "maximize":
            best_results = diff_treshold_df[diff_treshold_df['values_1'] == diff_treshold_df['values_1'].max()]
        else:
            best_results = diff_treshold_df[diff_treshold_df['values_1'] == diff_treshold_df['values_1'].min()]
        best_ind = best_results.index[0]
        params = study.get_trials()[best_ind].params

    display(best_results)

    return params


def plot_corrmatrix(df: pd.DataFrame,
                    drop_last: int=None,
                    calc_det: bool=False,
                    method: str='pearson'
                    ) -> None:

    MATRIX_SIZE = df.shape[1]

    if drop_last is None:
        corr_matrix = df.corr()

    else:
        corr_matrix = df.iloc[:, :-drop_last].corr(method)

    fig = plt.figure(figsize=(MATRIX_SIZE, MATRIX_SIZE))
    sns.heatmap(corr_matrix,annot=True,cmap=[ 'blue','lightblue','white','pink', 'red', 'darkred', 'black'])
    if calc_det:
        c_m_rank = np.linalg.matrix_rank(corr_matrix)
        c_m_det = np.linalg.det(corr_matrix)

        print(f'Ранг корреляционной матрицы = {c_m_rank} | Размер корреляционной матрицы {corr_matrix.shape[0]}')
        print(f'Детерминант корреляционной матрицы = {c_m_det:.3f}')


def get_total_data(X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    total_train = pd.concat([X_train, y_train], axis=1)
    total_test = pd.concat([X_test, y_test], axis=1)
    total_data = pd.concat([total_train, total_test], axis=0)
    return total_data


def get_VIF(df: pd.DataFrame) -> pd.DataFrame:

    df_corr = df.corr()
    df_corr['VIF'] = [variance_inflation_factor(df_corr.values, col_indx) for col_indx in range(df_corr.shape[1])]

    display(df_corr[["VIF"]].sort_values(by="VIF", ascending=False).T)


def get_test_metrics(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    trained_model: BaseEstimator) -> pd.DataFrame:

    TEST_ALL = pd.concat([ X_test, X_train])
    TARGET_ALL = pd.concat([ y_test, y_train])

    result_test, y_pred_test = get_prediction(X_test, y_test,
                                            trained_model)

    result_all, y_pred_all = get_prediction(TEST_ALL, TARGET_ALL,
                                            trained_model)

    y_pred_df = predicted_to_df(TARGET_ALL, y_pred_all)

    display(result_test)
    display(result_all)

    return y_pred_df, result_test, result_all


def get_pca(X_train: pd.DataFrame, X_test: pd.DataFrame,
            columns_pca: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    X_train = X_train.copy()
    X_test = X_test.copy()

    for column_name, columns_lst in columns_pca.items():
        pca = PCA(n_components=1)

        train_arr = pca.fit_transform(X_train.loc[:, columns_lst])
        test_arr = pca.transform(X_test.loc[:, columns_lst])

        X_train[column_name] = train_arr
        X_test[column_name] = test_arr

        X_train = X_train.drop(columns=columns_lst)
        X_test = X_test.drop(columns=columns_lst)

    return X_train, X_test


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


def plot_optuna_trained(study: optuna.study.study.Study,
                        value_0: str,
                        value_1: str,
                        params_importance: bool=False) -> None:

    pareto = optuna.visualization.plot_pareto_front(study, target_names=[value_0, value_1])
    slice_diff = optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[0], target_name=value_0)
    slice_r2 = optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[1], target_name=value_1)

    if params_importance:
        optuna.visualization.plot_param_importances(study)

    pareto.show()
    slice_diff.show()
    slice_r2.show()


def get_train_metrics(X_trian: pd.DataFrame, y_train: pd.Series,
                        model: BaseEstimator,
                        study: optuna.study.study.Study=None,
                        threshold: float=0.11,
                        get_coefs: bool=False,
                        data_name: str=None,
                        direction:Literal["minimize", "maximize"]="maximize",
                        cv_func:Literal[train_loo_cv, train_KF_cv, train_TS_cv]=train_loo_cv
                        ) -> BaseEstimator:

    if study is not None:
        params_threshold = get_best_study_params(study, threshold=threshold, direction=direction)
        trained_model, res = cv_func(X_trian, y_train,
                                        model.set_params(**params_threshold),
                                        get_coefs=get_coefs,
                                        data_name=data_name)

        display(pd.DataFrame([res]))

        return trained_model, pd.DataFrame([res])

    trained_model, res = cv_func(X_trian, y_train,
                                    model,
                                    get_coefs=get_coefs,
                                    data_name=data_name)

    display(pd.DataFrame([res]))

    return trained_model, pd.DataFrame([res])


def get_final_metrics(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        trained_model: BaseEstimator,
                        df_path: str,
                        map_dict: Dict[str, Tuple[str, Dict[int, str]]],
                        rename_col: str,
                        passflow_type: str,
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    TARGET_ALL = pd.concat([ y_test, y_train])
    TEST_ALL = pd.concat([ X_test, X_train])
    TEST_ALL.index = TARGET_ALL.index

    y_pred_df, result_test, result_all = get_test_metrics(X_train, y_train,
                                X_test, y_test,
                                trained_model)

    description_df = make_description_df(TARGET_ALL, y_pred_df,
                                        df_path,
                                        map_dict,
                                        rename_col,
                                        passflow_type,
                                        set_test_column=y_test)

    return description_df, TEST_ALL, TARGET_ALL, result_test, result_all


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


def get_optimized_metrics(
                        data_list: List[Tuple[str, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]],
                        base_model: BaseEstimator,
                        optimizer_func: Any,
                        optimizer_func_params: Dict[str, Any],
                        df_path: str = None,
                        map_dict: Dict[Any, Any] = None,
                        rename_col: Dict[Any, Any] = None,
                        passflow_type: str = None,
                        threshold: float=0.11,
                        direction: Literal["minimize", "maximize"]="maximize",
                        cv_func:Literal[train_loo_cv, train_KF_cv, train_TS_cv]=train_loo_cv
                        ) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    result_dict = {}
    data_names = []

    prog_bar_data = tqdm(data_list, total=len(data_list))
    if df_path is None:
        for data_name, X_train, y_train, X_test, y_test in prog_bar_data:
            data_names.append(data_name)

            params, study = optimizer_func(X_train, y_train, X_test,
                                            **optimizer_func_params,
                                            direction=direction,
                                            cv_func=cv_func
                                            )

            trained_model, train_metrics = get_train_metrics(X_train, y_train,
                                                            base_model,
                                                            study,
                                                            data_name=data_name,
                                                            threshold=threshold,
                                                            direction=direction,
                                                            cv_func=cv_func
                                                            )
            result_dict[data_name] = (params, study, trained_model, train_metrics)
        df_train_metrics = concat_metrics(result_dict, 3, data_names)
        # clear_output()

        return result_dict, df_train_metrics
    for data_name, X_train, y_train, X_test, y_test in prog_bar_data:
        data_names.append(data_name)

        params, study = optimizer_func(X_train, y_train, X_test,
                                        **optimizer_func_params,
                                        direction=direction,
                                        cv_func=cv_func
                                        )

        trained_model, train_metrics = get_train_metrics(X_train, y_train,
                                                        base_model,
                                                        study,
                                                        data_name=data_name,
                                                        threshold=threshold,
                                                        direction=direction,
                                                        cv_func=cv_func
                                                        )

        description_df, TEST_ALL, TARGET_ALL, test_metrics, all_metrics =  get_final_metrics(X_train, y_train,
                                                                                            X_test, y_test,
                                                                                            trained_model,
                                                                                            df_path=df_path,
                                                                                            map_dict=map_dict,
                                                                                            rename_col=rename_col,
                                                                                            passflow_type=passflow_type,
                                                                                            )

        result_dict[data_name] = (params, study, trained_model, train_metrics, test_metrics, all_metrics)

    df_train_metrics = concat_metrics(result_dict, 3, data_names)
    df_test_metrics = concat_metrics(result_dict, 4, data_names)
    df_all_metrics = concat_metrics(result_dict, 5, data_names)
    clear_output()

    return result_dict, df_train_metrics, df_test_metrics, df_all_metrics, description_df


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


def get_duplicated_df(df: pd.DataFrame, column: str) -> pd.DataFrame:

    dupl_df = df.duplicated(subset=[column])

    dupl_values = df.loc[dupl_df, column].values
    dupl_values_set = set(dupl_values)

    result_df = df[df[column].isin(dupl_values_set)]

    result_df = result_df.sort_values(by=column)
    return result_df


def get_polyfeatures(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    degree: int=2) -> Tuple[ pd.DataFrame, pd.DataFrame]:

    col_names = list(X_train.columns)

    poly = PolynomialFeatures(degree=degree, include_bias = False)
    poly_fit = poly.fit(X_train)
    X_train_poly_arr = poly_fit.transform(X_train)
    X_test_poly_arr = poly_fit.transform(X_test)

    df_columns = poly_fit.get_feature_names_out(col_names)

    X_train_poly = pd.DataFrame(X_train_poly_arr, columns=df_columns)
    X_test_poly = pd.DataFrame(X_test_poly_arr, columns=df_columns)

    return X_train_poly, X_test_poly


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
