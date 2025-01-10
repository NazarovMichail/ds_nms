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


def get_optimize_XGB(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"],
                    n_trials: int=100,
                    threshold=0.11,
                    learning_rate_range: Tuple[float]=(5e-4, 0.1),
                    verbosity_range: int=0,
                    tree_method_range: Tuple[str]=('exact', 'approx', 'hist'),
                    max_depth_range: Tuple[int]=(1, 5),
                    min_child_weight_range: Tuple[int]=(1, 40),
                    subsample_range: Tuple[float]=(0.1, 1),
                    gamma_range: Tuple[float]=(0, 2),
                    reg_lambda_range: Tuple[float]=(0, 4),
                    reg_alpha_range: Tuple[float]=(0, 2),
                    colsample_bytree_range: Tuple[float]=(0, 1),
                    model_name="model_name",
                    val_size: int=None
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize

    learning_rate_range=learning_rate_range
    verbosity_range=verbosity_range
    tree_method_range=tree_method_range
    max_depth_range=max_depth_range
    gamma_range=gamma_range
    min_child_weight_range=min_child_weight_range
    subsample_range=subsample_range
    reg_lambda_range=reg_lambda_range
    reg_alpha_range=reg_alpha_range
    colsample_bytree_range=colsample_bytree_range

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", *learning_rate_range)
        verbosity  = trial.suggest_categorical('verbosity', [verbosity_range])
        tree_method  = trial.suggest_categorical('tree_method', [*tree_method_range])
        max_depth = trial.suggest_int("max_depth", *max_depth_range)
        gamma = trial.suggest_float("gamma", *gamma_range)
        min_child_weight  = trial.suggest_int("min_child_weight", *min_child_weight_range)
        subsample  = trial.suggest_float("subsample", *subsample_range)
        reg_lambda  =  trial.suggest_float("reg_lambda", *reg_lambda_range)
        reg_alpha  = trial.suggest_float("reg_alpha", *reg_alpha_range)
        colsample_bytree = trial.suggest_float("colsample_bytree", *colsample_bytree_range)
        # создание и обучение модели
        model = XGBRegressor(
                    verbosity =verbosity ,
                    learning_rate=learning_rate ,
                    tree_method=tree_method,
                    max_depth=max_depth,
                    gamma=gamma,
                    min_child_weight=min_child_weight ,
                    subsample=subsample,
                    reg_lambda=reg_lambda,
                    reg_alpha  = reg_alpha  ,
                    colsample_bytree=colsample_bytree,
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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study


def get_optimize_Quant(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"],
                    n_trials: int=100,
                    threshold=0.11,
                    quantile_range: Tuple[float]=(0.4, 0.6),
                    alpha_range: Tuple[float]=(1e-5, 1e-2),
                    solver_range: str='highs',
                    model_name="model_name"
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize
    quantile_range = quantile_range
    solver_range = solver_range
    def objective(trial):
        # предложение гиперпараметров
        alpha =trial.suggest_float("alpha", *alpha_range)
        quantile = trial.suggest_float("quantile", *quantile_range, step=0.001)
        solver = trial.suggest_categorical("solver", [solver_range])

        model = QuantileRegressor(
                    quantile=quantile,
                    alpha=alpha,
                    solver=solver,
                    )

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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study


def get_optimize_DT(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    diff_metrics: Literal["R2_val_mean", "RMSE_val_mean", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="r2_diff",
                    metrics: Literal["R2_val", "RMSE_val", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="R2_val",
                    direction: Literal["minimize", "maximize"]="maximize",
                    n_trials: int=100,
                    threshold=0.11,
                    criterion_range: Tuple[str]=('squared_error', 'friedman_mse', 'absolute_error', 'poisson'),
                    max_depth_range: Tuple[int]=(1, 12),
                    min_samples_split_range: Tuple[int]=(2, 5),
                    min_samples_leaf_range: Tuple[int]=(1, 5),
                    max_features_range: Tuple[float]=(0.1, 1),
                    min_impurity_decrease_range: Tuple[float]=(0, 0.2),
                    ccp_alpha_range: Tuple[float]=(0, 0.3),
                    model_name="model_name",
                    cv_func:Literal[train_loo_cv, train_KF_cv]=train_loo_cv
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:


    X_ALL = pd.concat([X_train, X_test])

    diff_metrics_dict = {}
    diff_metrics = diff_metrics
    metrics = metrics

    criterion_range = criterion_range
    max_depth_range = max_depth_range
    min_samples_split_range = min_samples_split_range
    min_samples_leaf_range = min_samples_leaf_range
    max_features_range = max_features_range
    min_impurity_decrease_range = min_impurity_decrease_range
    ccp_alpha_range = ccp_alpha_range

    def objective(trial):
        # предложение гиперпараметров

        criterion = trial.suggest_categorical("criterion", [*criterion_range])
        max_depth = trial.suggest_int("max_depth", *max_depth_range)
        min_samples_split = trial.suggest_int("min_samples_split", *min_samples_split_range)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", *min_samples_leaf_range)
        max_features = trial.suggest_float("max_features", *max_features_range)
        min_impurity_decrease = trial.suggest_float("min_impurity_decrease", *min_impurity_decrease_range)
        ccp_alpha = trial.suggest_float("ccp_alpha", *ccp_alpha_range)

        model = DecisionTreeRegressor(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    min_impurity_decrease=min_impurity_decrease,
                    ccp_alpha=ccp_alpha,
                    random_state=1
                    )

        mdl, res = cv_func(X_train, y_train, model, get_coefs=False)

        R2_test_mean = res['R2_val_mean']
        r2_train_mean = res['R2_train_mean']

        RMSE_test_mean = res['RMSE_val_mean']
        RMSE_train_mean = res['RMSE_train_mean']

        MAE_test_mean = res['MAE_val_mean']
        MAE_train_mean = res['MAE_train_mean']

        r2_diff = abs(r2_train_mean - R2_test_mean)
        r2_relative_diff = r2_diff / r2_train_mean
        RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
        MAE_diff = abs(MAE_train_mean - MAE_test_mean)
        RMSE_diff_rel = RMSE_diff / RMSE_train_mean
        MAE_diff_rel = MAE_diff / MAE_train_mean

        y_pred = mdl.predict(X_ALL)
        negative_count = (y_pred < 0).sum()

        res["negative_count"] = negative_count
        res["r2_diff"] = r2_relative_diff
        res["RMSE_diff"] = RMSE_diff_rel
        res["MAE_diff"] = MAE_diff_rel

        return res[diff_metrics], res[metrics]

    study = optuna.create_study(study_name="params_study",
                                directions=[ "minimize", direction],
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold, direction=direction)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{diff_metrics}", second_param=f"{metrics}", model_name=model_name)

    return params, study


    study = optuna.create_study(study_name="params_study",
                                directions=[ "minimize", direction],
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold, direction=direction)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{diff_metrics}", second_param=f"{metrics}", model_name=model_name)

    return params, study


def get_optimize_SGD(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"],
                    n_trials: int=100,
                    threshold=0.11,
                    alpha_range: Tuple[float]=(0, 2 ),
                    penalty_range: Tuple[str]=('l2', 'l1', 'elasticnet' ),
                    l1_ratio_range: Tuple[float]=( 0, 1),
                    epsilon_range: Tuple[float]=( 0, 2),
                    eta0_range: Tuple[float]=(1e-4, 1 ),
                    power_t_range: Tuple[float]=(0, 2 ),
                    validation_fraction_range: Tuple[float]=(0.1, 1 ),
                    n_iter_no_change_range: Tuple[int]=(1, 5 ),
                    max_iter_range: int=150000,
                    model_name="model_name"
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize

    alpha_range=alpha_range
    penalty_range=penalty_range
    l1_ratio_range=l1_ratio_range
    epsilon_range=epsilon_range
    eta0_range=eta0_range
    power_t_range=power_t_range
    validation_fraction_range=validation_fraction_range
    n_iter_no_change_range=n_iter_no_change_range
    max_iter_range=max_iter_range

    def objective(trial):
        # предложение гиперпараметров
        alpha = trial.suggest_float("alpha", *alpha_range, log=False)
        penalty =trial.suggest_categorical("penalty", [*penalty_range])
        l1_ratio = trial.suggest_float("l1_ratio", *l1_ratio_range, log=False)
        epsilon = trial.suggest_float("epsilon", *epsilon_range, log=False)
        eta0 = trial.suggest_float("eta0", *eta0_range, log=False)
        power_t = trial.suggest_float("power_t", *power_t_range, log=False)
        validation_fraction = trial.suggest_float("validation_fraction", *validation_fraction_range, log=False)
        n_iter_no_change = trial.suggest_int("n_iter_no_change",*n_iter_no_change_range, log=False)
        max_iter =trial.suggest_categorical("max_iter", [max_iter_range])


        model = SGDRegressor(
                    alpha=alpha,
                    penalty=penalty,
                    l1_ratio=l1_ratio,
                    epsilon=epsilon,
                    eta0=eta0,
                    power_t=power_t,
                    validation_fraction=validation_fraction,
                    n_iter_no_change=n_iter_no_change,
                    max_iter=max_iter,
                    random_state=1
                    )

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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study


def get_optimize_RANSAC(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"], n_trials: int=100, threshold=0.11, model_name="model_name") -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize
    def objective(trial):
        # предложение гиперпараметров
        min_samples = trial.suggest_int("min_samples", 10, 100)
        residual_threshold = trial.suggest_int("residual_threshold", 100, 10000, step=100)
        loss = trial.suggest_categorical("loss", ['absolute_error'])
        C = trial.suggest_int("C", 100, 5000, step=100)
        kernel = trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf'])
        degree = trial.suggest_int("degree", 1, 3)
        coef0 = trial.suggest_float("coef0", 0, 1, log=False)
        epsilon = trial.suggest_float("epsilon", 0.1, 2, log=False)

        model = RANSACRegressor(
                    estimator=SVR(kernel=kernel, C=C, degree=degree, coef0=coef0, epsilon=epsilon),
                    min_samples=min_samples,
                    residual_threshold=residual_threshold,
                    loss=loss,
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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study


def get_optimize_Ridge(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame,
                        diff_metrics: Literal["R2_val_mean", "RMSE_val", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="r2_diff",
                        metrics: Literal["R2_val_mean", "RMSE_val", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="R2_val_mean",
                        direction: Literal["minimize", "maximize"]="maximize",
                        n_trials: int=100,
                        threshold=0.11,
                        solver_range: Tuple[str]=('svd', 'saga', 'lsqr'),
                        alpha_range: Tuple[float]=(0, 10),
                        max_iter_range: int=150000,
                        model_name="model_name",
                        cv_func:Literal[train_loo_cv, train_KF_cv]=train_loo_cv,
                        val_size: int=None,
                        data_name: str=None
                        ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    solver_range=solver_range
    alpha_range=alpha_range
    max_iter_range=max_iter_range
    def objective(trial):
        # предложение гиперпараметров
        solver = trial.suggest_categorical("solver", [*solver_range])
        alpha = trial.suggest_float("alpha", *alpha_range, log=False)
        max_iter = trial.suggest_categorical("max_iter", [max_iter_range])

        model = Ridge(
                    alpha=alpha,
                    solver=solver,
                    max_iter=max_iter,
                    positive=False,
                    random_state=1
                    )

        if val_size:
            result = {}
            X_train_val, X_val = X_train[:-val_size], X_train[-val_size:]
            y_train_val, y_val = y_train[:-val_size], y_train[-val_size:]

            mdl = model
            mdl.fit(X_train_val, y_train_val)

            result_train, y_train_pred = get_prediction(X_train_val, y_train_val, mdl)
            result_val, y_val_pred = get_prediction(X_val, y_val, mdl)

            R2_train = result_train['R2']
            RMSE_train = result_train['RMSE']
            NRMSE_train = result_train['NRMSE']
            MAE_train = result_train['MAE']
            RE_train = result_train['RE']

            R2_val = result_val['R2']
            RMSE_val = result_val['RMSE']
            NRMSE_val = result_val['NRMSE']
            MAE_val = result_val['MAE']
            RE_val = result_val['RE']

            result['R2_train'] = np.array(R2_train)
            result['RMSE_train'] = np.array(RMSE_train)
            result['NRMSE_train'] = np.array(NRMSE_train)
            result['MAE_train'] = np.array(MAE_train)
            result['RE_train'] = np.array(RE_train)

            result['R2_val'] = np.array(R2_val)
            result['RMSE_val'] = np.array(RMSE_val)
            result['NRMSE_val'] = np.array(NRMSE_val)
            result['MAE_val'] = np.array(MAE_val)
            result['RE_val'] = np.array(RE_val)

            final_result = {}
            final_result['model'] = f"{model}"

            for name, metric in result.items():
                final_result[f'{name}_mean'] = result[name].round(2)

            final_result['data_name'] = data_name

            R2_test_mean = final_result['R2_val_mean']
            r2_train_mean = final_result['R2_train_mean']

            RMSE_test_mean = final_result['RMSE_val_mean']
            RMSE_train_mean = final_result['RMSE_train_mean']

            MAE_test_mean = final_result['MAE_val_mean']
            MAE_train_mean = final_result['MAE_train_mean']

            r2_diff = abs(r2_train_mean - R2_test_mean)
            r2_relative_diff = r2_diff / r2_train_mean
            RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
            MAE_diff = abs(MAE_train_mean - MAE_test_mean)
            RMSE_diff_rel = RMSE_diff / RMSE_train_mean
            MAE_diff_rel = MAE_diff / MAE_train_mean

            y_pred = mdl.predict(X_ALL)
            negative_count = (y_pred < 0).sum()

            final_result["negative_count"] = negative_count
            final_result["r2_diff"] = r2_relative_diff
            final_result["RMSE_diff"] = RMSE_diff_rel
            final_result["MAE_diff"] = MAE_diff_rel

            return final_result[diff_metrics], final_result[metrics]

        mdl, res = cv_func(X_train, y_train, model, get_coefs=False)

        R2_test_mean = res['R2_val_mean']
        r2_train_mean = res['R2_train_mean']

        RMSE_test_mean = res['RMSE_val_mean']
        RMSE_train_mean = res['RMSE_train_mean']

        MAE_test_mean = res['MAE_val_mean']
        MAE_train_mean = res['MAE_train_mean']

        r2_diff = abs(r2_train_mean - R2_test_mean)
        r2_relative_diff = r2_diff / r2_train_mean
        RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
        MAE_diff = abs(MAE_train_mean - MAE_test_mean)
        RMSE_diff_rel = RMSE_diff / RMSE_train_mean
        MAE_diff_rel = MAE_diff / MAE_train_mean

        y_pred = mdl.predict(X_ALL)
        negative_count = (y_pred < 0).sum()

        res["negative_count"] = negative_count
        res["r2_diff"] = r2_relative_diff
        res["RMSE_diff"] = RMSE_diff_rel
        res["MAE_diff"] = MAE_diff_rel

        return res[diff_metrics], res[metrics]

    study = optuna.create_study(study_name="params_study",
                                directions=[ "minimize", direction],
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold, direction=direction)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{diff_metrics}", second_param=f"{metrics}", model_name=model_name)


    return params, study


def get_optimize_Lasso(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame,
                        diff_metrics: Literal["R2_val_mean", "RMSE_val_mean", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="r2_diff",
                        metrics: Literal["R2_val_mean", "RMSE_val_mean", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="R2_val_mean",
                        direction: Literal["minimize", "maximize"]="maximize",
                        n_trials: int=100,
                        threshold=0.11,
                        alpha_range: Tuple[float]=(0, 10),
                        max_iter_range: int=150000,
                        model_name="model_name",
                        cv_func:Literal[train_loo_cv, train_KF_cv, train_TS_cv]=train_loo_cv,
                        val_size: int=None,
                        data_name: str=None
                        ) -> Tuple[dict, Any, optuna.study.study.Study]:

    diff_metrics_dict = {}
    X_ALL = pd.concat([X_train, X_test])
    diff_metrics = diff_metrics
    metrics = metrics

    alpha_range=alpha_range
    max_iter_range=max_iter_range
    def objective(trial):
        # предложение гиперпараметров

        alpha = trial.suggest_float("alpha", *alpha_range, log=False)
        max_iter =trial.suggest_categorical("max_iter", [max_iter_range])

        model = Lasso(
                    alpha=alpha,
                    max_iter=max_iter,
                    positive=False,
                    random_state=1
                    )
        if val_size:
            result = {}
            X_train_val, X_val = X_train[:-val_size], X_train[-val_size:]
            y_train_val, y_val = y_train[:-val_size], y_train[-val_size:]

            mdl = model
            mdl.fit(X_train_val, y_train_val)

            result_train, y_train_pred = get_prediction(X_train_val, y_train_val, mdl)
            result_val, y_val_pred = get_prediction(X_val, y_val, mdl)

            R2_train = result_train['R2']
            RMSE_train = result_train['RMSE']
            NRMSE_train = result_train['NRMSE']
            MAE_train = result_train['MAE']
            RE_train = result_train['RE']

            R2_val = result_val['R2']
            RMSE_val = result_val['RMSE']
            NRMSE_val = result_val['NRMSE']
            MAE_val = result_val['MAE']
            RE_val = result_val['RE']

            result['R2_train'] = np.array(R2_train)
            result['RMSE_train'] = np.array(RMSE_train)
            result['NRMSE_train'] = np.array(NRMSE_train)
            result['MAE_train'] = np.array(MAE_train)
            result['RE_train'] = np.array(RE_train)

            result['R2_val'] = np.array(R2_val)
            result['RMSE_val'] = np.array(RMSE_val)
            result['NRMSE_val'] = np.array(NRMSE_val)
            result['MAE_val'] = np.array(MAE_val)
            result['RE_val'] = np.array(RE_val)

            final_result = {}
            final_result['model'] = f"{model}"

            for name, metric in result.items():
                final_result[f'{name}_mean'] = result[name].round(2)

            final_result['data_name'] = data_name

            R2_test_mean = final_result['R2_val_mean']
            r2_train_mean = final_result['R2_train_mean']

            RMSE_test_mean = final_result['RMSE_val_mean']
            RMSE_train_mean = final_result['RMSE_train_mean']

            MAE_test_mean = final_result['MAE_val_mean']
            MAE_train_mean = final_result['MAE_train_mean']

            r2_diff = abs(r2_train_mean - R2_test_mean)
            r2_relative_diff = r2_diff / r2_train_mean
            RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
            MAE_diff = abs(MAE_train_mean - MAE_test_mean)
            RMSE_diff_rel = RMSE_diff / RMSE_train_mean
            MAE_diff_rel = MAE_diff / MAE_train_mean

            y_pred = mdl.predict(X_ALL)
            negative_count = (y_pred < 0).sum()

            final_result["negative_count"] = negative_count
            final_result["r2_diff"] = r2_relative_diff
            final_result["RMSE_diff"] = RMSE_diff_rel
            final_result["MAE_diff"] = MAE_diff_rel

            return final_result[diff_metrics], final_result[metrics]

        mdl, res = cv_func(X_train, y_train, model, get_coefs=False)

        R2_test_mean = res['R2_val_mean']
        r2_train_mean = res['R2_train_mean']

        RMSE_test_mean = res['RMSE_val_mean']
        RMSE_train_mean = res['RMSE_train_mean']

        MAE_test_mean = res['MAE_val_mean']
        MAE_train_mean = res['MAE_train_mean']

        r2_diff = abs(r2_train_mean - R2_test_mean)
        r2_relative_diff = r2_diff / r2_train_mean
        RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
        MAE_diff = abs(MAE_train_mean - MAE_test_mean)
        RMSE_diff_rel = RMSE_diff / RMSE_train_mean
        MAE_diff_rel = MAE_diff / MAE_train_mean

        y_pred = mdl.predict(X_ALL)
        negative_count = (y_pred < 0).sum()

        res["negative_count"] = negative_count
        res["r2_diff"] = r2_relative_diff
        res["RMSE_diff"] = RMSE_diff_rel
        res["MAE_diff"] = MAE_diff_rel

        return res[diff_metrics], res[metrics]

    study = optuna.create_study(study_name="params_study",
                                directions=[ "minimize", direction],
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold, direction=direction)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{diff_metrics}", second_param=f"{metrics}", model_name=model_name)

    return params, study


def get_optimize_EN(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame,
                        diff_metrics: Literal["R2_val_mean", "RMSE_val", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="r2_diff",
                        metrics: Literal["R2_val", "RMSE_val", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="R2_val_mean",
                        direction: Literal["minimize", "maximize"]="maximize",
                        n_trials: int=100,
                        threshold=0.11,
                        l1_ratio_range: Tuple[float]=(0, 1),
                        alpha_range: Tuple[float]=(0, 10),
                        max_iter_range: int=150000,
                        model_name="model_name",
                        cv_func:Literal[train_loo_cv, train_KF_cv]=train_loo_cv,
                        data_name=None,
                        val_size: int=None
                        ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])

    diff_metrics_dict = {}
    diff_metrics = diff_metrics
    metrics = metrics

    l1_ratio_range = l1_ratio_range
    alpha_range = alpha_range
    max_iter_range  =max_iter_range
    def objective(trial):
        # предложение гиперпараметров
        l1_ratio = trial.suggest_float("l1_ratio", *l1_ratio_range)
        alpha = trial.suggest_float("alpha", *alpha_range, log=False)
        max_iter =trial.suggest_categorical("max_iter", [max_iter_range])

        model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                    positive=False,
                    random_state=1
                    )
        if val_size:
            result = {}
            X_train_val, X_val = X_train[:-val_size], X_train[-val_size:]
            y_train_val, y_val = y_train[:-val_size], y_train[-val_size:]

            mdl = model
            mdl.fit(X_train_val, y_train_val)

            result_train, y_train_pred = get_prediction(X_train_val, y_train_val, mdl)
            result_val, y_val_pred = get_prediction(X_val, y_val, mdl)

            R2_train = result_train['R2']
            RMSE_train = result_train['RMSE']
            NRMSE_train = result_train['NRMSE']
            MAE_train = result_train['MAE']
            RE_train = result_train['RE']

            R2_val = result_val['R2']
            RMSE_val = result_val['RMSE']
            NRMSE_val = result_val['NRMSE']
            MAE_val = result_val['MAE']
            RE_val = result_val['RE']

            result['R2_train'] = np.array(R2_train)
            result['RMSE_train'] = np.array(RMSE_train)
            result['NRMSE_train'] = np.array(NRMSE_train)
            result['MAE_train'] = np.array(MAE_train)
            result['RE_train'] = np.array(RE_train)

            result['R2_val'] = np.array(R2_val)
            result['RMSE_val'] = np.array(RMSE_val)
            result['NRMSE_val'] = np.array(NRMSE_val)
            result['MAE_val'] = np.array(MAE_val)
            result['RE_val'] = np.array(RE_val)

            final_result = {}
            final_result['model'] = f"{model}"

            for name, metric in result.items():
                final_result[f'{name}_mean'] = result[name].round(2)

            final_result['data_name'] = data_name

            R2_test_mean = final_result['R2_val_mean']
            r2_train_mean = final_result['R2_train_mean']

            RMSE_test_mean = final_result['RMSE_val_mean']
            RMSE_train_mean = final_result['RMSE_train_mean']

            MAE_test_mean = final_result['MAE_val_mean']
            MAE_train_mean = final_result['MAE_train_mean']

            r2_diff = abs(r2_train_mean - R2_test_mean)
            r2_relative_diff = r2_diff / r2_train_mean
            RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
            MAE_diff = abs(MAE_train_mean - MAE_test_mean)
            RMSE_diff_rel = RMSE_diff / RMSE_train_mean
            MAE_diff_rel = MAE_diff / MAE_train_mean

            y_pred = mdl.predict(X_ALL)
            negative_count = (y_pred < 0).sum()

            final_result["negative_count"] = negative_count
            final_result["r2_diff"] = r2_relative_diff
            final_result["RMSE_diff"] = RMSE_diff_rel
            final_result["MAE_diff"] = MAE_diff_rel

            return final_result[diff_metrics], final_result[metrics]

        mdl, res = cv_func(X_train, y_train, model, get_coefs=False)

        R2_test_mean = res['R2_val_mean']
        r2_train_mean = res['R2_train_mean']

        RMSE_test_mean = res['RMSE_val_mean']
        RMSE_train_mean = res['RMSE_train_mean']

        MAE_test_mean = res['MAE_val_mean']
        MAE_train_mean = res['MAE_train_mean']

        r2_diff = abs(r2_train_mean - R2_test_mean)
        r2_relative_diff = r2_diff / r2_train_mean
        RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
        MAE_diff = abs(MAE_train_mean - MAE_test_mean)
        RMSE_diff_rel = RMSE_diff / RMSE_train_mean
        MAE_diff_rel = MAE_diff / MAE_train_mean

        y_pred = mdl.predict(X_ALL)
        negative_count = (y_pred < 0).sum()

        res["negative_count"] = negative_count
        res["r2_diff"] = r2_relative_diff
        res["RMSE_diff"] = RMSE_diff_rel
        res["MAE_diff"] = MAE_diff_rel

        return res[diff_metrics], res[metrics]

    study = optuna.create_study(study_name="params_study",
                                directions=[ "minimize", direction],
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold, direction=direction)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{diff_metrics}", second_param=f"{metrics}", model_name=model_name)

    return params, study


def get_optimize_SVR(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"],
                    n_trials: int=100,
                    threshold=0.11,
                    kernel_range: Tuple[str]=('linear', 'poly', 'rbf'),
                    degree_range: Tuple[int]=(2, 3),
                    C_range: Tuple[int]= (0.01, 5000),
                    coef0_range: Tuple[float]=(0, 1),
                    epsilon_range: Tuple[float]=(0.1, 2),
                    gamma_range: Tuple[float]=(0.005, 0.5),
                    max_iter_range : int=150000,
                    model_name="model_name"
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize
    kernel_range = kernel_range
    degree_range=degree_range
    C_range=C_range
    coef0_range=coef0_range
    epsilon_range=epsilon_range
    gamma_range=gamma_range
    max_iter_range=max_iter_range

    def objective(trial):
        # предложение гиперпараметров
        kernel =trial.suggest_categorical("kernel", [*kernel_range])
        degree = trial.suggest_int("degree", *degree_range)
        C = trial.suggest_float("C", *C_range)
        coef0 = trial.suggest_float("coef0", *coef0_range, log=False)
        epsilon = trial.suggest_float("epsilon", *epsilon_range, log=False)
        gamma = trial.suggest_float("gamma", *gamma_range)
        max_iter = trial.suggest_categorical("max_iter", [max_iter_range])

        model = SVR(
                    kernel=kernel,
                    degree=degree,
                    coef0=coef0,
                    C=C,
                    epsilon=epsilon,
                    gamma=gamma,
                    max_iter=max_iter
                    )

        mdl, res = train_loo_cv(X_train, y_train, model, get_coefs=False)

        R2_test_mean = res['R2_val']
        r2_train_mean = res['R2_train_mean']

        RMSE_test_mean = res['RMSE_val']
        RMSE_train_mean = res['RMSE_train_mean']

        MAE_test_mean = res['MAE_val_mean']
        MAE_train_mean = res['MAE_train_mean']

        NRMSE_train_mean = res['NRMSE_train_mean']
        NRMSE_val = res['NRMSE_val']

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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study


def get_optimize_Huber(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame,
                        diff_metrics: Literal["R2_val_mean", "RMSE_val", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="r2_diff",
                        metrics: Literal["R2_val_mean", "RMSE_val", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="R2_val_mean",
                        direction: Literal["minimize", "maximize"]="maximize",
                        n_trials: int=100,
                        threshold=0.11,
                        epsilon_range: Tuple[float]=(1, 3),
                        alpha_range: Tuple[float]=(1e-5, 1e-1),
                        max_iter_range: int=150000,
                        model_name="model_name",
                        cv_func:Literal[train_loo_cv, train_KF_cv]=train_loo_cv,
                        val_size: int=None,
                        data_name=None
                        ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])

    diff_metrics_dict = {}

    epsilon_range = epsilon_range
    alpha_range = alpha_range
    max_iter_range = max_iter_range

    def objective(trial):
        # предложение гиперпараметров
        epsilon = trial.suggest_float("epsilon", *epsilon_range)
        alpha = trial.suggest_float("alpha", *alpha_range)
        max_iter = trial.suggest_categorical("max_iter", [max_iter_range])

        model = HuberRegressor(
                    epsilon=epsilon,
                    alpha=alpha,
                    max_iter=max_iter
                    )
        if val_size:
            result = {}
            X_train_val, X_val = X_train[:-val_size], X_train[-val_size:]
            y_train_val, y_val = y_train[:-val_size], y_train[-val_size:]

            mdl = model
            mdl.fit(X_train_val, y_train_val)

            result_train, y_train_pred = get_prediction(X_train_val, y_train_val, mdl)
            result_val, y_val_pred = get_prediction(X_val, y_val, mdl)

            R2_train = result_train['R2']
            RMSE_train = result_train['RMSE']
            NRMSE_train = result_train['NRMSE']
            MAE_train = result_train['MAE']
            RE_train = result_train['RE']

            R2_val = result_val['R2']
            RMSE_val = result_val['RMSE']
            NRMSE_val = result_val['NRMSE']
            MAE_val = result_val['MAE']
            RE_val = result_val['RE']

            result['R2_train'] = np.array(R2_train)
            result['RMSE_train'] = np.array(RMSE_train)
            result['NRMSE_train'] = np.array(NRMSE_train)
            result['MAE_train'] = np.array(MAE_train)
            result['RE_train'] = np.array(RE_train)

            result['R2_val'] = np.array(R2_val)
            result['RMSE_val'] = np.array(RMSE_val)
            result['NRMSE_val'] = np.array(NRMSE_val)
            result['MAE_val'] = np.array(MAE_val)
            result['RE_val'] = np.array(RE_val)

            final_result = {}
            final_result['model'] = f"{model}"

            for name, metric in result.items():
                final_result[f'{name}_mean'] = result[name].round(2)

            final_result['data_name'] = data_name

            R2_test_mean = final_result['R2_val_mean']
            r2_train_mean = final_result['R2_train_mean']

            RMSE_test_mean = final_result['RMSE_val_mean']
            RMSE_train_mean = final_result['RMSE_train_mean']

            MAE_test_mean = final_result['MAE_val_mean']
            MAE_train_mean = final_result['MAE_train_mean']

            r2_diff = abs(r2_train_mean - R2_test_mean)
            r2_relative_diff = r2_diff / r2_train_mean
            RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
            MAE_diff = abs(MAE_train_mean - MAE_test_mean)
            RMSE_diff_rel = RMSE_diff / RMSE_train_mean
            MAE_diff_rel = MAE_diff / MAE_train_mean

            y_pred = mdl.predict(X_ALL)
            negative_count = (y_pred < 0).sum()

            final_result["negative_count"] = negative_count
            final_result["r2_diff"] = r2_relative_diff
            final_result["RMSE_diff"] = RMSE_diff_rel
            final_result["MAE_diff"] = MAE_diff_rel

            return final_result[diff_metrics], final_result[metrics]

        mdl, res = cv_func(X_train, y_train, model, get_coefs=False)

        R2_test_mean = res['R2_val_mean']
        r2_train_mean = res['R2_train_mean']

        RMSE_test_mean = res['RMSE_val_mean']
        RMSE_train_mean = res['RMSE_train_mean']

        MAE_test_mean = res['MAE_val_mean']
        MAE_train_mean = res['MAE_train_mean']

        r2_diff = abs(r2_train_mean - R2_test_mean)
        r2_relative_diff = r2_diff / r2_train_mean
        RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
        MAE_diff = abs(MAE_train_mean - MAE_test_mean)
        RMSE_diff_rel = RMSE_diff / RMSE_train_mean
        MAE_diff_rel = MAE_diff / MAE_train_mean

        y_pred = mdl.predict(X_ALL)
        negative_count = (y_pred < 0).sum()

        res["negative_count"] = negative_count
        res["r2_diff"] = r2_relative_diff
        res["RMSE_diff"] = RMSE_diff_rel
        res["MAE_diff"] = MAE_diff_rel

        return res[diff_metrics], res[metrics]

    study = optuna.create_study(study_name="params_study",
                                directions=[ "minimize", direction],
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold, direction=direction)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{diff_metrics}", second_param=f"{metrics}", model_name=model_name)

    return params, study


def get_optimize_PAR(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"],
                    n_trials: int=100,
                    threshold=0.11,
                    n_iter_no_change_range: Tuple[int]=(1, 5),
                    validation_fraction_range: Tuple[float]=(0., 1),
                    C_range: Tuple[float]=(0.1, 5),
                    epsilon_range: Tuple[float]=(0, 5),
                    max_iter_range: int=150000,
                    model_name="model_name"
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize

    n_iter_no_change_range=n_iter_no_change_range
    validation_fraction_range=validation_fraction_range
    C_range=C_range
    epsilon_range=epsilon_range
    max_iter_range=max_iter_range

    def objective(trial):
        # предложение гиперпараметров
        n_iter_no_change =trial.suggest_int("n_iter_no_change", *n_iter_no_change_range)
        validation_fraction = trial.suggest_float("validation_fraction", *validation_fraction_range, log=False)
        C = trial.suggest_float("C", *C_range, log=False)
        epsilon = trial.suggest_float("epsilon", *epsilon_range, log=False)
        max_iter = trial.suggest_categorical("max_iter", [max_iter_range])

        model = PassiveAggressiveRegressor(
                    C=C,
                    epsilon=epsilon,
                    validation_fraction=validation_fraction,
                    n_iter_no_change=n_iter_no_change,
                    max_iter=max_iter,
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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study

def get_optimize_TS(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"], n_trials: int=100, threshold=0.11, model_name="model_name") -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize
    def objective(trial):
        # предложение гиперпараметров
        max_subpopulation =trial.suggest_int("max_subpopulation", 10, 10000)

        model = TheilSenRegressor(
                    max_subpopulation=max_subpopulation,
                    random_state=1,
                    max_iter=100000)

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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study


def get_optimize_BR(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"],
                    n_trials: int=100,
                    threshold=0.11,
                    alpha_1_range: Tuple[float]=(1e-9, 5e-5),
                    alpha_2_range: Tuple[float]=(1e-9, 5e-5),
                    lambda_1_range: Tuple[float]=(1e-9, 5e-5),
                    lambda_2_range: Tuple[float]=(1e-9, 5e-5),
                    max_iter_range: int=500,
                    model_name="model_name"
                    ) -> Tuple[dict, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize

    alpha_1_range=alpha_1_range
    alpha_2_range=alpha_2_range
    lambda_1_range=lambda_1_range
    lambda_2_range=lambda_2_range
    max_iter_range=max_iter_range

    def objective(trial):
        # предложение гиперпараметров
        alpha_1 = trial.suggest_float("alpha_1", *alpha_1_range, log=False)
        alpha_2 = trial.suggest_float("alpha_2", *alpha_2_range, log=False)
        lambda_1 = trial.suggest_float("lambda_1", *lambda_1_range, log=False)
        lambda_2 = trial.suggest_float("lambda_2", *lambda_2_range, log=False)
        max_iter = trial.suggest_categorical("max_iter", [max_iter_range])

        model = BayesianRidge(
                    alpha_1=alpha_1,
                    alpha_2=alpha_2,
                    lambda_1=lambda_1,
                    lambda_2=lambda_2,
                    compute_score=True,
                    max_iter=max_iter
                    )

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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study


def get_optimize_LGBM(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    diff_metrics: Literal["R2_val_mean", "RMSE_val_mean","MAE_val_mean","negative_count", "r2_diff", "RMSE_diff","MAE_diff", "RE_val_mean"]="r2_diff",
                    metrics: Literal["R2_val_mean", "RMSE_val_mean","MAE_val_mean","negative_count", "r2_diff", "RMSE_diff","MAE_diff", "RE_val_mean"]="R2_val_mean",
                    direction: Literal["minimize", "maximize"]="maximize",
                    n_trials: int=100,
                    threshold=0.11,
                    num_leaves_range: Tuple[int]=(2, 50),
                    learning_rate_range: Tuple[float]=(5e-4, 0.1),
                    max_depth_range: Tuple[int]=(1, 5),
                    reg_alpha_range: Tuple[float]=(0, 2),
                    reg_lambda_range: Tuple[float]=(0, 4),
                    verbose_range: int=-1,
                    min_child_samples_range: Tuple[int]=(1, 40),
                    tree_learner_range: Tuple[str]=('serial', 'feature', 'data'),
                    min_data_in_leaf_range: Tuple[int]=(2, 40),
                    feature_fraction_range: Tuple[float]=(0.1, 1),
                    n_estimators_range: Tuple[int]=(10, 100),
                    model_name="model_name",
                    cv_func:Literal[train_loo_cv, train_KF_cv, train_TS_cv]=train_loo_cv,
                    val_size: int=None,
                    data_name: str=None
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    diff_metrics = diff_metrics
    metrics = metrics

    num_leaves_range=num_leaves_range
    learning_rate_range=learning_rate_range
    max_depth_range=max_depth_range
    reg_alpha_range=reg_alpha_range
    reg_lambda_range=reg_lambda_range
    verbose_range=verbose_range
    min_child_samples_range=min_child_samples_range
    tree_learner_range=tree_learner_range
    min_data_in_leaf_range=min_data_in_leaf_range
    feature_fraction_range=feature_fraction_range
    n_estimators_range=n_estimators_range

    def objective(trial):
        num_leaves  = trial.suggest_int("num_leaves", *num_leaves_range)
        learning_rate = trial.suggest_float("learning_rate", *learning_rate_range)
        max_depth = trial.suggest_int("max_depth", *max_depth_range)
        reg_alpha  = trial.suggest_float("reg_alpha", *reg_alpha_range)
        reg_lambda  =  trial.suggest_float("reg_lambda", *reg_lambda_range)
        verbose = trial.suggest_categorical('verbose', [verbose_range])
        min_child_samples  = trial.suggest_int("min_child_samples", *min_child_samples_range)
        tree_learner = trial.suggest_categorical('tree_learner', [*tree_learner_range])
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", *min_data_in_leaf_range)
        n_estimators = trial.suggest_int("n_estimators", *n_estimators_range)
        feature_fraction  =  trial.suggest_float("feature_fraction", *feature_fraction_range)
        # создание и обучение модели
        model = LGBMRegressor(
                    num_leaves=num_leaves ,
                    learning_rate=learning_rate ,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    reg_alpha  = reg_alpha  ,
                    reg_lambda=reg_lambda,
                    verbose=verbose,
                    min_child_samples=min_child_samples ,
                    tree_learner=tree_learner,
                    min_data_in_leaf=min_data_in_leaf,
                    feature_fraction=feature_fraction,
                    random_state=1)

        if val_size:
            result = {}
            X_train_val, X_val = X_train[:-val_size], X_train[-val_size:]
            y_train_val, y_val = y_train[:-val_size], y_train[-val_size:]

            mdl = model
            mdl.fit(X_train_val, y_train_val)

            result_train, y_train_pred = get_prediction(X_train_val, y_train_val, mdl)
            result_val, y_val_pred = get_prediction(X_val, y_val, mdl)

            R2_train = result_train['R2']
            RMSE_train = result_train['RMSE']
            NRMSE_train = result_train['NRMSE']
            MAE_train = result_train['MAE']
            RE_train = result_train['RE']

            R2_val = result_val['R2']
            RMSE_val = result_val['RMSE']
            NRMSE_val = result_val['NRMSE']
            MAE_val = result_val['MAE']
            RE_val = result_val['RE']

            result['R2_train'] = np.array(R2_train)
            result['RMSE_train'] = np.array(RMSE_train)
            result['NRMSE_train'] = np.array(NRMSE_train)
            result['MAE_train'] = np.array(MAE_train)
            result['RE_train'] = np.array(RE_train)

            result['R2_val'] = np.array(R2_val)
            result['RMSE_val'] = np.array(RMSE_val)
            result['NRMSE_val'] = np.array(NRMSE_val)
            result['MAE_val'] = np.array(MAE_val)
            result['RE_val'] = np.array(RE_val)

            final_result = {}
            final_result['model'] = f"{model}"

            for name, metric in result.items():
                final_result[f'{name}_mean'] = result[name].round(2)

            final_result['data_name'] = data_name

            R2_test_mean = final_result['R2_val_mean']
            r2_train_mean = final_result['R2_train_mean']

            RMSE_test_mean = final_result['RMSE_val_mean']
            RMSE_train_mean = final_result['RMSE_train_mean']

            MAE_test_mean = final_result['MAE_val_mean']
            MAE_train_mean = final_result['MAE_train_mean']

            r2_diff = abs(r2_train_mean - R2_test_mean)
            r2_relative_diff = r2_diff / r2_train_mean
            RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
            MAE_diff = abs(MAE_train_mean - MAE_test_mean)
            RMSE_diff_rel = RMSE_diff / RMSE_train_mean
            MAE_diff_rel = MAE_diff / MAE_train_mean

            y_pred = mdl.predict(X_ALL)
            negative_count = (y_pred < 0).sum()

            final_result["negative_count"] = negative_count
            final_result["r2_diff"] = r2_relative_diff
            final_result["RMSE_diff"] = RMSE_diff_rel
            final_result["MAE_diff"] = MAE_diff_rel

            return final_result[diff_metrics], final_result[metrics]

        mdl, res = cv_func(X_train, y_train, model, get_coefs=False)

        R2_test_mean = res['R2_val_mean']
        r2_train_mean = res['R2_train_mean']

        RMSE_test_mean = res['RMSE_val_mean']
        RMSE_train_mean = res['RMSE_train_mean']

        MAE_test_mean = res['MAE_val_mean']
        MAE_train_mean = res['MAE_train_mean']

        r2_diff = abs(r2_train_mean - R2_test_mean)
        r2_relative_diff = r2_diff / r2_train_mean
        RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
        MAE_diff = abs(MAE_train_mean - MAE_test_mean)
        RMSE_diff_rel = RMSE_diff / RMSE_train_mean
        MAE_diff_rel = MAE_diff / MAE_train_mean

        y_pred = mdl.predict(X_ALL)
        negative_count = (y_pred < 0).sum()

        res["negative_count"] = negative_count
        res["r2_diff"] = r2_relative_diff
        res["RMSE_diff"] = RMSE_diff_rel
        res["MAE_diff"] = MAE_diff_rel

        return res[diff_metrics], res[metrics]

    study = optuna.create_study(study_name="params_study",
                                directions=[ "minimize", direction],
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold, direction=direction)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{diff_metrics}", second_param=f"{metrics}", model_name=model_name)

    return params, study


def get_optimize_ARD(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"],
                    n_trials: int=100,
                    threshold=0.11,
                    max_iter_range: int=150000,
                    alpha_1_range: Tuple[float]=(1e-9, 5e-5),
                    alpha_2_range: Tuple[float]=(1e-9, 5e-5),
                    lambda_1_range: Tuple[float]=(1e-9, 5e-5),
                    lambda_2_range: Tuple[float]=(1e-9, 5e-5),
                    model_name="model_name"
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize

    max_iter_range=max_iter_range
    alpha_1_range=alpha_1_range
    alpha_2_range=alpha_2_range
    lambda_1_range=lambda_1_range
    lambda_2_range=lambda_2_range

    def objective(trial):
        # предложение гиперпараметров
        alpha_1 = trial.suggest_float("alpha_1", *alpha_1_range, log=False)
        alpha_2 = trial.suggest_float("alpha_2", *alpha_2_range, log=False)
        lambda_1 = trial.suggest_float("lambda_1", *lambda_1_range, log=False)
        lambda_2 = trial.suggest_float("lambda_2", *lambda_2_range, log=False)
        max_iter = trial.suggest_categorical("max_iter", [max_iter_range])

        model = ARDRegression(
                    max_iter=max_iter,
                    alpha_1=alpha_1,
                    alpha_2=alpha_2,
                    lambda_1=lambda_1,
                    lambda_2=lambda_2,
                    )

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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study


def get_optimize_RF(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"],
                    n_trials: int=100,
                    threshold=0.11,
                    max_features_range: Tuple[str, int]=('sqrt', 'log2', 1),
                    criterion_range: Tuple[str, int]=('squared_error', 'absolute_error', 'friedman_mse', 'poisson'),
                    n_estimators_range: Tuple[int]=(50, 150),
                    max_depth_range: Tuple[int]=(1, 10),
                    max_leaf_nodes_range: Tuple[int]=(5, 15),
                    min_weight_fraction_leaf_range: Tuple[float]=(0, 0.5),
                    model_name="model_name"
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize

    max_features_range=max_features_range
    criterion_range=criterion_range
    n_estimators_range=n_estimators_range
    max_depth_range=max_depth_range
    max_leaf_nodes_range=max_leaf_nodes_range
    min_weight_fraction_leaf_range=min_weight_fraction_leaf_range

    def objective(trial):

        max_features = trial.suggest_categorical('max_features', [*max_features_range])
        criterion = trial.suggest_categorical('criterion', [*criterion_range])
        n_estimators = trial.suggest_int("n_estimators", *n_estimators_range)
        max_depth = trial.suggest_int("max_depth", *max_depth_range)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", *max_leaf_nodes_range)
        min_weight_fraction_leaf  =  trial.suggest_float("min_weight_fraction_leaf", *min_weight_fraction_leaf_range)

        # создание и обучение модели
        model = RandomForestRegressor(
                    n_estimators=n_estimators ,
                    criterion=criterion ,
                    max_depth=max_depth,
                    min_weight_fraction_leaf =min_weight_fraction_leaf ,
                    max_features=max_features,
                    verbose=0,
                    max_leaf_nodes=max_leaf_nodes ,
                    oob_score=True,
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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study


def get_optimize_ET(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"],
                    n_trials: int=100,
                    threshold=0.11,
                    max_features_range: Tuple[str]=('sqrt', 'log2'),
                    criterion_range: Tuple[str]=('squared_error', 'friedman_mse', 'poisson'),
                    max_depth_range: Tuple[int]=(1, 7),
                    max_leaf_nodes_range: Tuple[int]=(5, 20),
                    min_weight_fraction_leaf_range: Tuple[float]=(0, 0.2),
                    model_name="model_name"
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize

    max_features_range=max_features_range
    criterion_range=criterion_range
    max_depth_range=max_depth_range
    max_leaf_nodes_range=max_leaf_nodes_range
    min_weight_fraction_leaf_range=min_weight_fraction_leaf_range

    def objective(trial):

        max_features = trial.suggest_categorical('max_features', [*max_features_range])
        criterion = trial.suggest_categorical('criterion', [*criterion_range])
        max_depth = trial.suggest_int("max_depth", *max_depth_range)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", *max_leaf_nodes_range)
        min_weight_fraction_leaf  =  trial.suggest_float("min_weight_fraction_leaf", *min_weight_fraction_leaf_range)

        # создание и обучение модели
        model = ExtraTreesRegressor(
                    criterion=criterion ,
                    max_depth=max_depth,
                    min_weight_fraction_leaf =min_weight_fraction_leaf ,
                    max_features=max_features,
                    max_leaf_nodes=max_leaf_nodes ,
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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study


def get_optimize_KNN(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    diff_metrics: Literal["R2_val", "RMSE_val", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="r2_diff",
                    metrics: Literal["R2_val", "RMSE_val", "MAE_val_mean","negative_count", "r2_diff", "RMSE_diff", "MAE_diff", "RE_val_mean"]="R2_val",
                    direction: Literal["minimize", "maximize"]="maximize",
                    n_trials: int=100,
                    threshold=0.11,
                    weights_range: Tuple[str]=('uniform', 'distance'),
                    n_neighbors_range: Tuple[int]=(1, 7),
                    p_range: Tuple[int]=(1, 3),
                    model_name="model_name",
                    cv_func:Literal[train_loo_cv, train_KF_cv]=train_loo_cv
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])

    diff_metrics_dict = {}
    diff_metrics = diff_metrics
    metrics = metrics

    n_neighbors_range=n_neighbors_range
    weights_range=weights_range
    p_range = p_range

    def objective(trial):

        n_neighbors = trial.suggest_int("n_neighbors", *n_neighbors_range)
        weights = trial.suggest_categorical('weights', [*weights_range])
        p = trial.suggest_int("p", *p_range)

        # создание и обучение модели
        model = KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    p=p
                    )

        mdl, res = cv_func(X_train, y_train, model, get_coefs=False)

        R2_test_mean = res['R2_val_mean']
        r2_train_mean = res['R2_train_mean']

        RMSE_test_mean = res['RMSE_val_mean']
        RMSE_train_mean = res['RMSE_train_mean']

        MAE_test_mean = res['MAE_val_mean']
        MAE_train_mean = res['MAE_train_mean']

        r2_diff = abs(r2_train_mean - R2_test_mean)
        r2_relative_diff = r2_diff / r2_train_mean
        RMSE_diff = abs(RMSE_train_mean - RMSE_test_mean)
        MAE_diff = abs(MAE_train_mean - MAE_test_mean)
        RMSE_diff_rel = RMSE_diff / RMSE_train_mean
        MAE_diff_rel = MAE_diff / MAE_train_mean

        y_pred = mdl.predict(X_ALL)
        negative_count = (y_pred < 0).sum()

        res["negative_count"] = negative_count
        res["r2_diff"] = r2_relative_diff
        res["RMSE_diff"] = RMSE_diff_rel
        res["MAE_diff"] = MAE_diff_rel

        return res[diff_metrics], res[metrics]

    study = optuna.create_study(study_name="params_study",
                                directions=[ "minimize", direction],
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold, direction=direction)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{diff_metrics}", second_param=f"{metrics}", model_name=model_name)

    return params, study


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


def get_optimize_Stack_EN(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    estimators_dict: Dict[str, BaseEstimator],
                    minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"],
                    n_trials: int=100,
                    threshold=0.11,
                    l1_ratio_range: Tuple[float]=(0, 1),
                    alpha_range: Tuple[float]=(0, 2),
                    max_iter_range: int=150000,
                    model_name="model_name"
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize

    estimators_dict = estimators_dict
    l1_ratio_range=l1_ratio_range
    alpha_range=alpha_range
    max_iter_range=max_iter_range

    def objective(trial):
        # предложение гиперпараметров
        l1_ratio = trial.suggest_float("l1_ratio", *l1_ratio_range)
        alpha = trial.suggest_float("alpha", *alpha_range, log=False)
        max_iter =trial.suggest_categorical("max_iter", [max_iter_range])

        model = StackingRegressor(
                    list(estimators_dict.items()),
                    ElasticNet(
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                            max_iter=max_iter,
                            positive=False,
                            random_state=1
                        )
                    )

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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study


def get_optimize_TR(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame,
                    minimize: Literal["negative_count", "r2_diff", "RMSE_diff", "MAE_diff"],
                    n_trials: int=100,
                    threshold=0.11,
                    power_range: Tuple[float]=(0, 3),
                    alpha_range: Tuple[float]=(0, 2),
                    max_iter_range: int=1000,
                    model_name="model_name"
                    ) -> Tuple[dict, Any, optuna.study.study.Study]:

    X_ALL = pd.concat([X_train, X_test])
    minimize = minimize

    power_range = power_range
    alpha_range = alpha_range
    max_iter_range = max_iter_range

    def objective(trial):
        # предложение гиперпараметров
        power = trial.suggest_float("power", *power_range, step=0.01)
        alpha = trial.suggest_float("alpha", *alpha_range)
        max_iter = trial.suggest_categorical("max_iter", [max_iter_range])

        model = TweedieRegressor(
                    power=power,
                    alpha=alpha,
                    max_iter=max_iter
                    )

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
                                sampler=optuna.samplers.TPESampler(seed=1),
                                pruner=optuna.pruners.HyperbandPruner()
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    trial = study.best_trials

    params = get_best_study_params(study=study, threshold=threshold)

    clear_output()
    print("Parameters updated !")

    optuna_plot(study, dir="params_plots/", param_importances=False, first_param=f"{minimize}", second_param="R2_test_mean", model_name=model_name)

    return params, study
