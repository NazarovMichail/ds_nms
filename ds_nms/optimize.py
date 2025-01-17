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
from funcs import *


np.random.seed(1)




def mlflow_train(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, models_dict: Dict[str, BaseEstimator], exp_name: str=None, n_trials: int=300) -> pd.DataFrame:

    # Установка эксперимента
    if exp_name is not None:
        experiment_id = mlflow.create_experiment(exp_name, artifact_location=f"s3://bucket-nazarovmichail//{exp_name}")
        mlflow.set_experiment(exp_name)

    runs_dict = {}
    with mlflow.start_run(run_name=f"{exp_name}_run") as run:
        for model_name in models_dict.keys():
            with mlflow.start_run(run_name=f"{model_name}_{exp_name}", nested=True) as child_run:
                X_all = pd.concat( [X_train, X_test] )
                y_all = pd.concat( [y_train, y_test] )
                if model_name == "SVR_rbf":
                    params, study = get_optimize_TR(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=TweedieRegressor(**params),
                                                get_coefs=False)

                if model_name == "SVR_rbf":
                    params, study = get_optimize_SVR(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=SVR(**params),
                                                get_coefs=False)
                if model_name == "HuberRegressor":
                    params, study = get_optimize_Huber(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=HuberRegressor(**params),
                                                get_coefs=False)
                if model_name == "PassiveAggressiveRegressor":
                    params, study = get_optimize_PAR(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=PassiveAggressiveRegressor(**params, random_state=1),
                                                get_coefs=False)
                if model_name == "TheilSenRegressor":
                    params, study = get_optimize_TS(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=TheilSenRegressor(**params, random_state=1),
                                                get_coefs=False)

                if model_name == "BayesianRidge":
                    params, study = get_optimize_BR(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=BayesianRidge(**params),
                                                get_coefs=False)
                if model_name == "LGBMRegressor":
                    params, study = get_optimize_LGBM(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=LGBMRegressor(**params, random_state=1),
                                                get_coefs=False)
                if model_name == "ARDRegression":
                    params, study = get_optimize_ARD(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=ARDRegression(**params),
                                                get_coefs=False)
                if model_name == "Ridge":
                    params, study = get_optimize_Ridge(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=Ridge(**params, random_state=1),
                                                get_coefs=False)
                if model_name == "Lasso":
                    params, study = get_optimize_Lasso(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=Lasso(**params, random_state=1),
                                                get_coefs=False)
                if model_name == "ElasticNet":
                    params, study = get_optimize_EN(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=ElasticNet(**params, random_state=1),
                                                get_coefs=False)
                if model_name == "DecisionTreeRegressor":
                    params, study = get_optimize_DT(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=DecisionTreeRegressor(**params, random_state=1),
                                                get_coefs=False)
                if model_name == "KNeighborsRegressor":
                    params, study = get_optimize_KNN(X_train, y_train, X_test, minimize='r2_diff', n_trials=n_trials, model_name=model_name)
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=KNeighborsRegressor(**params),
                                                get_coefs=False)
                else:
                    model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=models_dict[model_name],
                                                get_coefs=False)

                result_test_dict, y_pred = get_prediction(X_test, y_test, model_trained)
                result_all_dict, y_all_pred = get_prediction(X_all, y_all, model_trained)

                r2_train = result_train_dict['R2_train_mean']
                rmse_train = result_train_dict['RMSE_train_mean']
                mae_train = result_train_dict['MAE_train_mean']
                nrmse_train = result_train_dict['NRMSE_train_mean']

                r2_val = result_train_dict['R2_val']
                rmse_val = result_train_dict['RMSE_val']
                mae_val = result_train_dict['MAE_val_mean']
                nrmse_val = result_train_dict['NRMSE_val']

                r2_test = result_test_dict['R2']
                rmse_test = result_test_dict['RMSE']
                mae_test = result_test_dict['MAE']
                nrmse_test = result_test_dict['NRMSE']

                negatives = result_all_dict['negative']
                re = (y_all_pred - y_all).abs()
                re_relative = (re / y_all) * 100
                max_re = re_relative.max()
                count_less_30 = re_relative[ re_relative < 30].shape[0]
                re_less_30 = count_less_30 / y_all.shape[0]

                mlflow.log_metric("1. r2_train", r2_train)
                mlflow.log_metric("2. r2_val", r2_val)
                mlflow.log_metric("3. r2_test", r2_test)
                mlflow.log_metric("4. nrmse_train", nrmse_train)
                mlflow.log_metric("5. nrmse_val", nrmse_val)
                mlflow.log_metric("6. nrmse_test", nrmse_test)
                mlflow.log_metric("7. negatives", negatives)
                mlflow.log_metric("8. re_less_30", re_less_30)
                mlflow.log_metric("9. max_re", max_re)
                mlflow.log_metric("mae_train", mae_train)
                mlflow.log_metric("mae_test", mae_test)
                mlflow.log_metric("mae_val", mae_val)

                # signature = infer_signature(X_train, y_train)
                model_info = mlflow.sklearn.log_model(model_trained, f"{model_name}_{exp_name}")

                y_pred_df = predicted_to_df(y_all, y_all_pred)

                runs_dict[model_name] = child_run
                runs_dict[f"{model_name}_y_pred"] = y_pred_df

    clear_output()
    return runs_dict, study


def mlflow_train_base(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, models_dict: Dict[str, BaseEstimator], exp_name: str=None) -> pd.DataFrame:

    # Установка эксперимента
    if exp_name is not None:
        experiment_id = mlflow.create_experiment(exp_name, artifact_location=f"s3://bucket-nazarovmichail//{exp_name}")
        mlflow.set_experiment(exp_name)

    runs_dict = {}
    with mlflow.start_run(run_name=f"{exp_name}_run") as run:
        for model_name in models_dict.keys():
            with mlflow.start_run(run_name=f"{model_name}_{exp_name}", nested=True) as child_run:
                X_all = pd.concat( [X_train, X_test] )
                y_all = pd.concat( [y_train, y_test] )

                model_trained, result_train_dict  = train_loo_cv(X=X_train,
                                                y=y_train,
                                                model=models_dict[model_name],
                                                get_coefs=False)

                result_test_dict, y_pred = get_prediction(X_test, y_test, model_trained)
                result_all_dict, y_all_pred = get_prediction(X_all, y_all, model_trained)

                r2_train = result_train_dict['R2_train_mean']
                rmse_train = result_train_dict['RMSE_train_mean']
                mae_train = result_train_dict['MAE_train_mean']
                nrmse_train = result_train_dict['NRMSE_train_mean']

                r2_val = result_train_dict['R2_val_mean']
                rmse_val = result_train_dict['RMSE_val_mean']
                mae_val = result_train_dict['MAE_val_mean']
                nrmse_val = result_train_dict['NRMSE_val_mean']

                r2_test = result_test_dict['R2']
                rmse_test = result_test_dict['RMSE']
                mae_test = result_test_dict['MAE']
                nrmse_test = result_test_dict['NRMSE']

                negatives = result_all_dict['negative']
                re = (y_all_pred - y_all).abs()
                re_relative = (re / y_all) * 100
                max_re = re_relative.max()
                count_less_30 = re_relative[ re_relative < 30].shape[0]
                re_less_30 = count_less_30 / y_all.shape[0]

                mlflow.log_metric("1. r2_train", r2_train)
                mlflow.log_metric("2. r2_val", r2_val)
                mlflow.log_metric("3. r2_test", r2_test)
                mlflow.log_metric("4. nrmse_train", nrmse_train)
                mlflow.log_metric("5. nrmse_val", nrmse_val)
                mlflow.log_metric("6. nrmse_test", nrmse_test)
                mlflow.log_metric("7. negatives", negatives)
                mlflow.log_metric("8. re_less_30", re_less_30)
                mlflow.log_metric("9. max_re", max_re)
                mlflow.log_metric("mae_train", mae_train)
                mlflow.log_metric("mae_test", mae_test)
                mlflow.log_metric("mae_val", mae_val)

                # signature = infer_signature(X_train, y_train)
                model_info = mlflow.sklearn.log_model(model_trained, f"{model_name}_{exp_name}")

                y_pred_df = predicted_to_df(y_all, y_all_pred)

                runs_dict[model_name] = child_run
                runs_dict[f"{model_name}_y_pred"] = y_pred_df
    clear_output()
    return runs_dict
