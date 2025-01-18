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


def get_total_data(X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    total_train = pd.concat([X_train, y_train], axis=1)
    total_test = pd.concat([X_test, y_test], axis=1)
    total_data = pd.concat([total_train, total_test], axis=0)
    return total_data





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
