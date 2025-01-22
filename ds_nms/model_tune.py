import pandas as pd
from typing import List, Tuple, Any, Dict, Literal
import pickle
import os
from sklearn.ensemble import IsolationForest, ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
import numpy as np
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator
import optuna
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PassiveAggressiveRegressor, LassoLars, BayesianRidge, HuberRegressor, QuantileRegressor, RANSACRegressor, TheilSenRegressor, PoissonRegressor, TweedieRegressor, ARDRegression, SGDRegressor, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate, StratifiedKFold, LeaveOneOut
from tqdm import tqdm
from IPython.display import clear_output
from  datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, kruskal
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer, MinMaxScaler
from IPython.display import display
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, median_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import mlflow
from mlflow.models import infer_signature
from ds_nms import model_train, utils_io


def get_best_study_params(study: optuna.study.study.Study,
                          threshold: float | int,
                          direction_1: Literal["minimize", "maximize"]="minimize",
                          direction_2: Literal["minimize", "maximize"]="maximize"):

    results_df = study.trials_dataframe()
    # удаление +/- inf из 'values_0' и 'values_1'
    results_df[['values_0', 'values_1']] = results_df[['values_0', 'values_1']].replace([np.inf, -np.inf], np.nan)
    results_df = results_df.dropna(subset=['values_0', 'values_1'])
    if results_df.empty:
        print("Нет валидных строк после удаления +/- inf из 'values_0' и 'values_1'.")
        return None

    if direction_2 == "minimize":
        filtered_df_1 = results_df[results_df['values_1'] < threshold]
    else:
        filtered_df_1 = results_df[results_df['values_1'] > threshold]

    if direction_1 == "maximize":
        best_results = filtered_df_1[filtered_df_1['values_0'] == filtered_df_1['values_0'].max()]
    else:
        best_results = filtered_df_1[filtered_df_1['values_0'] == filtered_df_1['values_0'].min()]
    try:
        best_ind = best_results.index[0]
        params = study.get_trials()[best_ind].params
    except IndexError as error:
        print(error)
        try:
            if direction_2 == "minimize":
                filtered_df_1 = results_df[results_df['values_1'] == results_df['values_1'].min() ]
            else:
                filtered_df_1 = results_df[results_df['values_1'] == results_df['values_1'].max()]
            if direction_1 == "maximize":
                best_results = filtered_df_1[filtered_df_1['values_0'] == filtered_df_1['values_0'].min()]
            else:
                best_results = filtered_df_1[filtered_df_1['values_0'] == filtered_df_1['values_0'].max()]

            best_ind = best_results.index[0]
            params = study.get_trials()[best_ind].params
        except Exception as e:
            display(best_results)
            print(e)
            print('''Проверьте метрики для оптимизации... \nДля loo подходят: R2_val_micro, RMSE_val_micro, MAE_val_micro ''')
            return None

    display(best_results)

    return params

def optuna_plot(study: optuna.study.study.Study,
                directory: str = None,
                param_importances: bool=False,
                name_metric_1: str="first_param",
                name_metric_2: str="second_param",
                model_name="model_name") -> None:

    if directory is not None:
        DATE = dt.now().strftime("%Y_%m_%d_%H_%M")

        os.makedirs(name=f"optuna_plots/{directory}", exist_ok=True)

        try:
            optuna.visualization.plot_pareto_front(study,
                                                   target_names=[f"{name_metric_1}",
                                                                 f"{name_metric_2}"]).write_html(f'optuna_plots/{directory}/{model_name}_pareto_front_{DATE}.html')
            optuna.visualization.plot_slice(study,
                                            target=lambda trial: trial.values[0],
                                            target_name=f'{name_metric_1}').write_html(f'optuna_plots/{directory}/{model_name}_metric_1_slice_{DATE}.html')
            optuna.visualization.plot_slice(study,
                                            target=lambda trial: trial.values[1],
                                            target_name=f'{name_metric_2}').write_html(f'optuna_plots/{directory}/{model_name}_metric_2_slice_{DATE}.html')

            if param_importances:
                optuna.visualization.plot_param_importances(study).write_html(f'optuna_plots/{directory}/param_importances_{DATE}.html')

        except Exception as e:
            print(e)
    else:
        optuna.visualization.plot_pareto_front(study, target_names=[f"{name_metric_1}",
                                                                    f"{name_metric_2}"]).show()
        optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[0],
                                        target_name=f'{name_metric_1}').show()
        optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[1],
                                        target_name=f'{name_metric_2}').show()

        if param_importances:
            optuna.visualization.plot_param_importances(study).show()

def extract_model_params(trial: optuna.Trial,
                        param_dict: dict) -> Tuple[dict, dict]:

    model_params_optim = {}
    model_params_base = {}
    for param_name, cfg in param_dict.items():
        if param_name == "base_params":
            model_params_base = cfg
        else:
            param_type = cfg["type"]
            args = cfg["args"]
            kwargs = cfg.get("kwargs", {})
            if param_type == "float":
                model_params_optim[param_name] = trial.suggest_float(param_name, *args, **kwargs)
            elif param_type == "int":
                model_params_optim[param_name] = trial.suggest_int(param_name, *args, **kwargs)
            elif param_type == "categorical":
                model_params_optim[param_name] = trial.suggest_categorical(param_name, *args, **kwargs)
            else:
                raise ValueError(f"Неизвестный тип параметра: {param_type}")
    return model_params_optim, model_params_base

def get_optimize_params(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame,
                        metric_1: Literal["R2_val_macro",
                                          "RMSE_val_macro",
                                          "MAE_val_macro",
                                          "RE_val_macro",
                                          "R2_val_micro",
                                          "RMSE_val_micro",
                                          "MAE_val_micro",
                                          "RE_val_micro",
                                          "negative_all",
                                          "R2_diff_rel",
                                          "RMSE_diff_rel",
                                          "MAE_diff_rel"]="R2_val_micro",
                        metric_2: Literal["R2_val_macro",
                                          "RMSE_val_macro",
                                          "MAE_val_macro",
                                          "RE_val_macro",
                                          "R2_val_micro",
                                          "RMSE_val_micro",
                                          "MAE_val_micro",
                                          "RE_val_micro",
                                          "negative_all",
                                          "R2_diff_rel",
                                          "RMSE_diff_rel",
                                          "MAE_diff_rel"]="RMSE_diff_rel",
                        direction_1: Literal["minimize", "maximize"]="maximize",
                        direction_2: Literal["minimize", "maximize"]="minimize",
                        n_trials: int=100,
                        threshold=0.11,
                        cv_type: Literal['kf', 'loo', 'stratify', 'ts']='loo',
                        metric_best: Literal['R2_val', 'RMSE_val', 'NRMSE_val', 'MAE_val', 'RE_val' ]='MAE_val',
                        n_splits: int = 5,
                        train_size: int = 48,
                        val_size: int = 12,
                        data_name: str=None,
                        model_cls: BaseEstimator = None,
                        model_params: Dict[str, dict] = None,
                        final_estim_cls: BaseEstimator = None # Только для Stacking
                        ) -> Tuple[dict, Any, optuna.study.study.Study]:

    if model_cls is None:
        raise ValueError("Не указан класс модели (model_cls).")
    if model_params is None:
        raise ValueError("Не передано описание пространства гиперпараметров (model_params).")

    X_all = pd.concat([X_train, X_test])

    def objective(trial):

        optim_params, base_params = extract_model_params(trial, model_params)
        if model_cls == StackingRegressor:
            base_params['final_estimator'] = final_estim_cls(**optim_params)
            optim_params = {}
        model = model_cls(**optim_params, **base_params)

        best_model, res_dict = model_train.train_cv(X=X_train, y=y_train,
                                               model=model,
                                               cv_type=cv_type,
                                               metric_best=metric_best,
                                               n_splits=n_splits,
                                               train_size=train_size,
                                               val_size=val_size)

        y_pred = best_model.predict(X_all)
        negative_all = (y_pred < 0).sum()
        res_dict["negative_all"] = negative_all

        return res_dict[metric_1], res_dict[metric_2]

    study = optuna.create_study(study_name="params_study",
                                directions=[ direction_1, direction_2],
                                sampler=optuna.samplers.TPESampler(seed=1, multivariate=True),
                                # pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)

    best_params = get_best_study_params(study=study,
                                   threshold=threshold,
                                   direction_1=direction_1,
                                   direction_2=direction_2)

    clear_output()


    optuna_plot(study=study,
                directory=data_name,
                param_importances=False,
                name_metric_1=f"{metric_1}",
                name_metric_2=f"{metric_2}",
                model_name=model_cls.__name__ )

    return best_params, study

def get_optimize_results(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        metric_1: Literal["R2_val_macro",
                                          "RMSE_val_macro",
                                          "MAE_val_macro",
                                          "RE_val_macro",
                                          "R2_val_micro",
                                          "RMSE_val_micro",
                                          "MAE_val_micro",
                                          "RE_val_micro",
                                          "negative_all",
                                          "R2_diff_rel",
                                          "RMSE_diff_rel",
                                          "MAE_diff_rel"]="R2_val_micro",
                        metric_2: Literal["R2_val_macro",
                                          "RMSE_val_macro",
                                          "MAE_val_macro",
                                          "RE_val_macro",
                                          "R2_val_micro",
                                          "RMSE_val_micro",
                                          "MAE_val_micro",
                                          "RE_val_micro",
                                          "negative_all",
                                          "R2_diff_rel",
                                          "RMSE_diff_rel",
                                          "MAE_diff_rel"]="RMSE_diff_rel",
                        direction_1: Literal["minimize", "maximize"]="maximize",
                        direction_2: Literal["minimize", "maximize"]="minimize",
                        n_trials: int=100,
                        threshold=0.11,
                        cv_type: Literal['kf', 'loo', 'stratify', 'ts']='loo',
                        metric_best: Literal['R2_val', 'RMSE_val', 'NRMSE_val', 'MAE_val', 'RE_val' ]='MAE_val',
                        show_plots: bool = True,
                        n_splits: int = 5,
                        train_size: int = 48,
                        val_size: int = 12,
                        data_name: str=None,
                        model_cls: BaseEstimator = None,
                        model_params: Dict[str, dict] = None,
                        final_estim_cls: BaseEstimator = None # Только для Stacking
                        ) -> Dict[str,
                                Tuple[BaseEstimator, Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] ]:

    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    #----------------------------------------------------#
    # Подбор оптимальных параметров модели
    #----------------------------------------------------#
    best_params, study = get_optimize_params(X_train=X_train, y_train=y_train, X_test=X_test,
                                            metric_1=metric_1,
                                            metric_2=metric_2,
                                            direction_1=direction_1,
                                            direction_2=direction_2,
                                            n_trials=n_trials,
                                            threshold=threshold,
                                            cv_type=cv_type,
                                            n_splits=n_splits,
                                            train_size=train_size,
                                            val_size=val_size,
                                            metric_best=metric_best,
                                            data_name=data_name,
                                            model_cls=model_cls,
                                            model_params=model_params)

    #----------------------------------------------------#
    # Обучение модели с подобранными параметрами
    #----------------------------------------------------#
    best_trained_model, cv_metrics = model_train.train_cv(X=X_train, y=y_train,
                                                    model=model_cls(**best_params),
                                                    cv_type=cv_type,
                                                    n_splits=n_splits,
                                                    train_size=train_size,
                                                    val_size=val_size,
                                                    metric_best=metric_best,
                                                    data_name=data_name)

    #----------------------------------------------------#
    # Получение метрик
    #----------------------------------------------------#
    train_metrics, _ = model_train.get_prediction(X=X_train, y=y_train,
                                                model=best_trained_model,
                                                data_name=data_name,
                                                metrics_type='train'
                                                        )
    test_metrics, _ = model_train.get_prediction(X=X_test, y=y_test,
                                                model=best_trained_model,
                                                data_name=data_name,
                                                metrics_type='test'
                                                        )
    all_metrics, y_pred_all = model_train.get_prediction(X=X_all, y=y_all,
                                                model=best_trained_model,
                                                data_name=data_name,
                                                metrics_type='all'
                                                        )
    #----------------------------------------------------#
    # Добавление целевых переменных в датафрейм признаков
    #----------------------------------------------------#
    y_pred_all.name = 'y_pred'
    y_all.name = 'y_true'

    pred_df = pd.concat([X_all , y_all, y_pred_all], axis=1)

    pred_df['abs_error'] = pred_df['y_pred'] - pred_df['y_true']
    pred_df['rel_error'] = round(abs(pred_df['abs_error'] / pred_df['y_true']), 3) * 100
    pred_df['test_data'] = " "
    pred_df.loc[y_test.index, 'test_data'] = "X"

    pred_df.sort_values(by='rel_error', ascending=False, inplace=True)

    optimize_results = {}
    optimize_results['model'] = best_trained_model
    optimize_results['study'] = study
    optimize_results['cv_metrics'] = pd.DataFrame([cv_metrics])
    optimize_results['train_metrics'] = train_metrics
    optimize_results['test_metrics'] = test_metrics
    optimize_results['all_metrics'] = all_metrics
    optimize_results['pred_df'] = pred_df

    #----------------------------------------------------#
    # Визуализация подбора параметров
    #----------------------------------------------------#
    if show_plots:
        optuna_plot(study=study,name_metric_1=metric_1, name_metric_2=metric_2)
        print(f'Ключи для получения результатов: {list(optimize_results.keys())}')

    return optimize_results

def get_optimize_several_results(data_dict: Dict[str, List[pd.DataFrame]],
                                models_dict: Dict[BaseEstimator, dict],
                                metric_1: Literal["R2_val_macro",
                                                  "RMSE_val_macro",
                                                  "MAE_val_macro",
                                                  "RE_val_macro",
                                                  "R2_val_micro",
                                                  "RMSE_val_micro",
                                                  "MAE_val_micro",
                                                  "RE_val_micro",
                                                  "negative_all",
                                                  "R2_diff_rel",
                                                  "RMSE_diff_rel",
                                                  "MAE_diff_rel"]="R2_val_micro",
                                metric_2: Literal["R2_val_macro",
                                                  "RMSE_val_macro",
                                                  "MAE_val_macro",
                                                  "RE_val_macro",
                                                  "R2_val_micro",
                                                  "RMSE_val_micro",
                                                  "MAE_val_micro",
                                                  "RE_val_micro",
                                                  "negative_all",
                                                  "R2_diff_rel",
                                                  "RMSE_diff_rel",
                                                  "MAE_diff_rel"]="RMSE_diff_rel",
                                direction_1: Literal["minimize", "maximize"]="maximize",
                                direction_2: Literal["minimize", "maximize"]="minimize",
                                n_trials: int=100,
                                threshold=0.11,
                                cv_type: Literal['kf', 'loo', 'stratify', 'ts']='loo',
                                metric_best: Literal['R2_val', 'RMSE_val', 'NRMSE_val', 'MAE_val', 'RE_val' ]='MAE_val',
                                n_splits: int = 5,
                                train_size: int = 48,
                                val_size: int = 12,
                                show_plots: bool = True
                                ) -> Dict[str, dict]:

    final_result_dict = {}
    prog_bar_data = tqdm(data_dict.items(), total=len(data_dict.keys()))
    for data_name, data_list in prog_bar_data:
        train_metrics_lst = []
        val_metrics_lst = []
        test_metrics_lst = []
        studies_lst = []
        for model_cls, params in models_dict.items():
            optim_result_dict = get_optimize_results(X_train=data_list[0], y_train=data_list[1],
                                                                X_test=data_list[2], y_test=data_list[3],
                                                                metric_1=metric_1,
                                                                metric_2=metric_2,
                                                                direction_1=direction_1,
                                                                direction_2=direction_2,
                                                                n_trials=n_trials,
                                                                threshold=threshold,
                                                                cv_type=cv_type,
                                                                metric_best=metric_best,
                                                                show_plots=False,
                                                                n_splits=n_splits,
                                                                train_size=train_size,
                                                                val_size=val_size,
                                                                data_name=data_name,
                                                                model_cls=model_cls,
                                                                model_params=params
                                                                )
            train_metrics_lst.append(optim_result_dict['train_metrics'])
            val_metrics_lst.append(optim_result_dict['cv_metrics'])
            test_metrics_lst.append(optim_result_dict['test_metrics'])
            studies_lst.append(optim_result_dict['study'])

        final_result_dict[data_name] = {"train_metrics": pd.concat(train_metrics_lst),
                                         "val_metrics": pd.concat(val_metrics_lst),
                                         "test_metrics": pd.concat(test_metrics_lst),
                                         "study": studies_lst}
    if show_plots:
        for data, res_dict in final_result_dict.items():
            for i, model_name in enumerate(models_dict.keys()):
                cur_study = res_dict['study'][i]
                print(f"Данные: {data} | Модель: {model_name.__name__}")
                optuna_plot(study=cur_study, name_metric_1=metric_1, name_metric_2=metric_2)

    print(f'Данные для обучения моделей: {list(final_result_dict.keys())}')
    print(f'Собраны метрики: {list(final_result_dict[data_name].keys())}')

    return final_result_dict

def mlflow_save_results(X: pd.DataFrame, y: pd.Series,
                        optim_results: Dict[str, dict],
                        data_name: str) -> None:
    """Логирование метрик и обученных моделей

    Args:
        X (pd.DataFrame): Обучающий датафрейм для примера
        y (pd.Series): Целевые переменные для примера
        optim_results (Dict[str, dict]): Словарь с результатами обучени моделей
        data_name (str): Название данных (название эксперимента)
    """

    #----------------------------------------------------#
    # Очистка логов mlflow, запуск сервара mlflow
    #----------------------------------------------------#
    utils_io.mlflow_run_delete()
    utils_io.mlflow_server_start()

    try:
        experiment_id = mlflow.create_experiment(name=data_name,
                                            artifact_location=f"mlruns/{data_name}/"
                                            )
        mlflow.set_experiment(experiment_id=experiment_id)
    except Exception:
        mlflow.set_experiment(experiment_name=data_name)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run(run_name=f"{data_name}_run") as run:
        for model_name, result_dict in optim_results.items():
            with mlflow.start_run(run_name=f"{model_name}_{data_name}", nested=True) as child_run:

                #----------------------------------------------------#
                # Логирование метрик
                #----------------------------------------------------#
                mlflow.log_metric("1. R2_train", result_dict['train_metrics']['R2_train'].values.round(3).item())
                mlflow.log_metric("2. R2_val", result_dict['cv_metrics']['R2_val_micro'].values.round(3).item())
                mlflow.log_metric("3. R2_test", result_dict['test_metrics']['R2_test'].values.round(3).item())
                mlflow.log_metric("4. NRMSE_train", result_dict['train_metrics']['NRMSE_train'].values.round(3).item())
                mlflow.log_metric("5. NRMSE_val", result_dict['cv_metrics']['NRMSE_val_micro'].values.round(3).item())
                mlflow.log_metric("6. NRMSE_test", result_dict['test_metrics']['NRMSE_test'].values.round(3).item())
                mlflow.log_metric("7. MAE_train", result_dict['train_metrics']['MAE_train'].values.round(3).item())
                mlflow.log_metric("8. MAE_val", result_dict['cv_metrics']['MAE_val_micro'].values.round(3).item())
                mlflow.log_metric("9. MAE_test", result_dict['test_metrics']['MAE_test'].values.round(3).item())
                mlflow.log_metric("Negatives", result_dict['all_metrics']['negative_all'].values.item())
                mlflow.log_metric("RE", result_dict['all_metrics']['RE_all'].values.round(2).item())
                #----------------------------------------------------#
                # Логирование модели
                #----------------------------------------------------#
                mlflow.sklearn.log_model(result_dict['model'],
                                         f"{model_name}_{data_name}",
                                         input_example=X)

def get_prediction_df(init_df: pd.DataFrame,
                      pred_df: pd.DataFrame,
                      init_cols: List[str],
                      pred_cols: List[str] = ['y_true', 'y_pred', 'abs_error', 'rel_error', 'test_data'],
                      sort_by: str = 'rel_error') -> pd.DataFrame:

    """Создание датафрейма с предсказаниями и описательными столбцами

    Args:
        init_df (pd.DataFrame): Исходный датафрейм
        pred_df (pd.DataFrame): Датафрейм с предсказаниями
        init_cols (List[str]): Имена столбцов исходного датафрейма для объединения
        pred_cols (List[str], optional): Имена столбцов датафрейма с предсказаниями для объединения. Defaults to ['y_true', 'y_pred', 'abs_error', 'rel_error', 'test_data'].
        sort_by (str, optional): Столбец по которому отсортировать датафрейм. Defaults to 'rel_error'.

    Returns:
        pd.DataFrame: Датафрейм с предсказаниями и описательными столбцами
    """
    init_df = init_df[init_cols]
    result_df = pd.concat([init_df, pred_df.loc[:, pred_cols]], axis=1)
    result_df.sort_values(by=sort_by, ascending=False, inplace=True)

    return result_df
