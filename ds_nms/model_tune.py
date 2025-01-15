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
from ds_nms import model_train


def get_best_study_params(study: optuna.study.study.Study,
                          threshold: float | int,
                          direction_1: Literal["minimize", "maximize"]="minimize",
                          direction_2: Literal["minimize", "maximize"]="maximize"):

    results_df = study.trials_dataframe()

    if direction_1 == "minimize":
        filtered_df_1 = results_df[results_df['values_0'] < threshold]
    else:
        filtered_df_1 = results_df[results_df['values_0'] > threshold]

    if direction_2 == "maximize":
        best_results = filtered_df_1[filtered_df_1['values_1'] == filtered_df_1['values_1'].max()]
    else:
        best_results = filtered_df_1[filtered_df_1['values_1'] == filtered_df_1['values_1'].min()]
    try:
        best_ind = best_results.index[0]
        params = study.get_trials()[best_ind].params
    except IndexError as error:
        print(error)
        if direction_1 == "minimize":
            filtered_df_1 = results_df[results_df['values_0'] == results_df['values_0'].min() ]
        else:
            filtered_df_1 = results_df[results_df['values_0'] == results_df['values_0'].max()]
        if direction_2 == "maximize":
            best_results = filtered_df_1[filtered_df_1['values_1'] == filtered_df_1['values_1'].min()]
        else:
            best_results = filtered_df_1[filtered_df_1['values_1'] == filtered_df_1['values_1'].max()]

        best_ind = best_results.index[0]
        params = study.get_trials()[best_ind].params

    display(best_results)

    return params

def optuna_plot(study: optuna.study.study.Study,
                directory: str,
                param_importances: bool=False,
                name_metric_1: str="first_param",
                name_metric_2: str="second_param",
                model_name="model_name") -> None:

    DATE = dt.now().strftime("%Y_%m_%d_%H_%M")

    os.makedirs(name="optuna_plots", exist_ok=True)

    try:
        optuna.visualization.plot_pareto_front(study, target_names=[f"{name_metric_1}", f"{name_metric_2}"]).write_html(f'optuna_plots/{directory}{model_name}_pareto_front_{DATE}.html')
        optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[0], target_name=f'{name_metric_1}').write_html(f'optuna_plots/{directory}{model_name}_diff_plot_slice_{DATE}.html')
        optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[1], target_name=f'{name_metric_2}').write_html(f'optuna_plots/{directory}{model_name}_metrics_plot_slice_{DATE}.html')

        if param_importances:
            optuna.visualization.plot_param_importances(study).write_html(f'optuna_plots/{directory}param_importances_{DATE}.html')

    except FileNotFoundError:
        dir_path = os.path.join("optuna_plots", f"{directory}")
        os.mkdir(dir_path)
        print("________________________________________________________________________________________________")
        print(f"Создана директория: {directory} для размещения графиков оптимизации гиперпараметров")
        optuna.visualization.plot_pareto_front(study, target_names=[f"{name_metric_1}", f"{name_metric_2}"]).write_html(f'optuna_plots/{directory}{model_name}_pareto_front_{DATE}.html')
        optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[0], target_name=f'{name_metric_1}').write_html(f'optuna_plots/{directory}{model_name}_diff_plot_slice_{DATE}.html')
        optuna.visualization.plot_slice(study,  target=lambda trial: trial.values[1], target_name=f'{name_metric_2}').write_html(f'optuna_plots/{directory}{model_name}_metrics_plot_slice_{DATE}.html')

        if param_importances:
            optuna.visualization.plot_param_importances(study).write_html(f'optuna_plots/{directory}{model_name}_param_importances_{DATE}.html')

def get_optimize_Lasso(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame,
                        metric_1: Literal["R2_val_mean", "RMSE_val_mean", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="R2_diff_rel",
                        metric_2: Literal["R2_val_mean", "RMSE_val_mean", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="R2_val_micro",
                        direction_1: Literal["minimize", "maximize"]="minimize",
                        direction_2: Literal["minimize", "maximize"]="maximize",
                        n_trials: int=100,
                        threshold=0.11,
                        cv_type: str = 'loo',
                        metric_best: Literal['R2_val', 'RMSE_val', 'NRMSE_val', 'MAE_val', 'RE_val' ]='MAE_val',
                        n_splits: int = 5,
                        train_size: int = 48,
                        val_size: int = 12,
                        alpha_range: Tuple[float]=(0, 1000),
                        max_iter_range: int=150000,
                        model_name="model_name",
                        data_name: str=None
                        ) -> Tuple[dict, Any, optuna.study.study.Study]:


    X_all = pd.concat([X_train, X_test])

    def objective(trial):

        alpha = trial.suggest_float("alpha", *alpha_range, log=False)
        max_iter =trial.suggest_categorical("max_iter", [max_iter_range])

        model = Lasso(
                    alpha=alpha,
                    max_iter=max_iter,
                    positive=False,
                    random_state=1
                    )

        model, res_dict = model_train.train_cv(X=X_train, y=y_train,
                                               model=model,
                                               cv_type=cv_type,
                                               metric_best=metric_best,
                                               n_splits=n_splits,
                                               train_size=train_size,
                                               val_size=val_size)

        y_pred = model.predict(X_all)
        negative_all = (y_pred < 0).sum()
        res_dict["negative_count"] = negative_all

        return res_dict[metric_1], res_dict[metric_2]

    study = optuna.create_study(study_name="params_study",
                                directions=[ direction_1, direction_2],
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study,
                                   threshold=threshold,
                                   direction_1=direction_1,
                                   direction_2=direction_2)

    clear_output()

    optuna_plot(study=study,
                directory="params_plots/",
                param_importances=False,
                name_metric_1=f"{metric_1}",
                name_metric_2=f"{metric_2}",
                model_name=model_name)

    return params, study
