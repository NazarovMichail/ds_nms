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


class Models_params():

    def __init__(self):
# $$\
# $$ |
# $$ |      $$$$$$\   $$$$$$$\  $$$$$$$\  $$$$$$\
# $$ |      \____$$\ $$  _____|$$  _____|$$  __$$\
# $$ |      $$$$$$$ |\$$$$$$\  \$$$$$$\  $$ /  $$ |
# $$ |     $$  __$$ | \____$$\  \____$$\ $$ |  $$ |
# $$$$$$$$\\$$$$$$$ |$$$$$$$  |$$$$$$$  |\$$$$$$  |
# \________|\_______|\_______/ \_______/  \______/

        self.lasso_params_optim = {
            "alpha": {
                "type": "float",
                "args": [0, 1000],
                "kwargs": {"log": False}
            },
            "base_params": {"positive": False,
                            "random_state": 1,
                            "max_iter": 150000}
        }
# $$$$$$$\  $$\       $$\
# $$  __$$\ \__|      $$ |
# $$ |  $$ |$$\  $$$$$$$ | $$$$$$\   $$$$$$\
# $$$$$$$  |$$ |$$  __$$ |$$  __$$\ $$  __$$\
# $$  __$$< $$ |$$ /  $$ |$$ /  $$ |$$$$$$$$ |
# $$ |  $$ |$$ |$$ |  $$ |$$ |  $$ |$$   ____|
# $$ |  $$ |$$ |\$$$$$$$ |\$$$$$$$ |\$$$$$$$\
# \__|  \__|\__| \_______| \____$$ | \_______|
#                         $$\   $$ |
#                         \$$$$$$  |
#                          \______/
        self.ridge_params_optim = {
            "alpha": {
                "type": "float",
                "args": [0, 1000],
                "kwargs": {"log": False}
            },
            "solver": {
                "type": "categorical",
                "args": [['svd', 'saga', 'lsqr']],
                "kwargs": {}
            },
        }
        self.ridge_params = {"positive": False,
                            "random_state": 1,
                            "max_iter": 150000}

# $$\   $$\           $$\
# $$ |  $$ |          $$ |
# $$ |  $$ |$$\   $$\ $$$$$$$\   $$$$$$\   $$$$$$\
# $$$$$$$$ |$$ |  $$ |$$  __$$\ $$  __$$\ $$  __$$\
# $$  __$$ |$$ |  $$ |$$ |  $$ |$$$$$$$$ |$$ |  \__|
# $$ |  $$ |$$ |  $$ |$$ |  $$ |$$   ____|$$ |
# $$ |  $$ |\$$$$$$  |$$$$$$$  |\$$$$$$$\ $$ |
# \__|  \__| \______/ \_______/  \_______|\__|

        self.huber_params_optim = {
            "alpha": {
                "type": "float",
                "args": [0, 1000],
                "kwargs": {"log": False}
            },
            "epsilon": {
                "type": "float",
                "args": [1, 3],
                "kwargs": {"log": False}
            },
            "base_params":{"max_iter": 150000}
        }

# $$$$$$$$\ $$\                       $$\     $$\           $$\   $$\            $$\
# $$  _____|$$ |                      $$ |    \__|          $$$\  $$ |           $$ |
# $$ |      $$ | $$$$$$\   $$$$$$$\ $$$$$$\   $$\  $$$$$$$\ $$$$\ $$ | $$$$$$\ $$$$$$\
# $$$$$\    $$ | \____$$\ $$  _____|\_$$  _|  $$ |$$  _____|$$ $$\$$ |$$  __$$\\_$$  _|
# $$  __|   $$ | $$$$$$$ |\$$$$$$\    $$ |    $$ |$$ /      $$ \$$$$ |$$$$$$$$ | $$ |
# $$ |      $$ |$$  __$$ | \____$$\   $$ |$$\ $$ |$$ |      $$ |\$$$ |$$   ____| $$ |$$\
# $$$$$$$$\ $$ |\$$$$$$$ |$$$$$$$  |  \$$$$  |$$ |\$$$$$$$\ $$ | \$$ |\$$$$$$$\  \$$$$  |
# \________|\__| \_______|\_______/    \____/ \__| \_______|\__|  \__| \_______|  \____/

        self.elasticnet_params_optim = {
            "alpha": {
                "type": "float",
                "args": [0, 1000],
                "kwargs": {"log": False}
            },
            "l1_ratio": {
                "type": "float",
                "args": [0, 1],
                "kwargs": {"log": False}
            },
            "base_params":{"positive": False,
                            "random_state": 1,
                            "max_iter": 150000}
        }

# $$\   $$\  $$$$$$\  $$$$$$$\
# $$ |  $$ |$$  __$$\ $$  __$$\
# \$$\ $$  |$$ /  \__|$$ |  $$ |
#  \$$$$  / $$ |$$$$\ $$$$$$$\ |
#  $$  $$<  $$ |\_$$ |$$  __$$\
# $$  /\$$\ $$ |  $$ |$$ |  $$ |
# $$ /  $$ |\$$$$$$  |$$$$$$$  |
# \__|  \__| \______/ \_______/

        self.xgb_params_optim = {
            "learning_rate": {
                "type": "float",
                "args": [5e-4, 0.1],
                "kwargs": {"log": False}
            },
            "tree_method": {
                "type": "categorical",
                "args": [['exact', 'approx', 'hist']],
                "kwargs": {}
            },
            "max_depth": {
                "type": "int",
                "args": [1, 5],
                "kwargs": {}
            },
            "gamma": {
                "type": "float",
                "args": [0, 2],
                "kwargs": {"log": False}
            },
            "min_child_weight": {
                "type": "int",
                "args": [1, 40],
                "kwargs": {}
            },
            "subsample": {
                "type": "float",
                "args": [0.1, 1],
                "kwargs": {"log": False}
            },
            "reg_lambda": {
                "type": "float",
                "args": [0, 4],
                "kwargs": {"log": False}
            },
            "reg_alpha": {
                "type": "float",
                "args": [0, 1],
                "kwargs": {"log": False}
            },
            "colsample_bytree": {
                "type": "float",
                "args": [0, 1],
                "kwargs": {"log": False}
            },
            "base_params":{"random_state": 1,
                           "verbosity": 0}
        }

#  $$$$$$\                                  $$\     $$\ $$\
# $$  __$$\                                 $$ |    \__|$$ |
# $$ /  $$ |$$\   $$\  $$$$$$\  $$$$$$$\  $$$$$$\   $$\ $$ | $$$$$$\
# $$ |  $$ |$$ |  $$ | \____$$\ $$  __$$\ \_$$  _|  $$ |$$ |$$  __$$\
# $$ |  $$ |$$ |  $$ | $$$$$$$ |$$ |  $$ |  $$ |    $$ |$$ |$$$$$$$$ |
# $$ $$\$$ |$$ |  $$ |$$  __$$ |$$ |  $$ |  $$ |$$\ $$ |$$ |$$   ____|
# \$$$$$$ / \$$$$$$  |\$$$$$$$ |$$ |  $$ |  \$$$$  |$$ |$$ |\$$$$$$$\
#  \___$$$\  \______/  \_______|\__|  \__|   \____/ \__|\__| \_______|
#      \___|

        self.quantile_params_optim = {
            "alpha": {
                "type": "float",
                "args": [0, 1000],
                "kwargs": {"log": False}
            },
            "solver": {
                "type": "categorical",
                "args": [['highs', 'highs-ds']],
                "kwargs": {}
            },
            "quantile": {
                "type": "float",
                "args": [0.4, 0.6],
                "kwargs": {"log": False}
            },
            "base_params":{}
        }

# $$$$$$$\  $$$$$$$$\
# $$  __$$\ \__$$  __|
# $$ |  $$ |   $$ |    $$$$$$\   $$$$$$\   $$$$$$\
# $$ |  $$ |   $$ |   $$  __$$\ $$  __$$\ $$  __$$\
# $$ |  $$ |   $$ |   $$ |  \__|$$$$$$$$ |$$$$$$$$ |
# $$ |  $$ |   $$ |   $$ |      $$   ____|$$   ____|
# $$$$$$$  |   $$ |   $$ |      \$$$$$$$\ \$$$$$$$\
# \_______/    \__|   \__|       \_______| \_______|

        self.dt_params_optim = {
            "criterion": {
                "type": "categorical",
                "args": [['squared_error', 'friedman_mse', 'absolute_error', 'poisson']],
                "kwargs": {}
            },
            "max_depth": {
                "type": "int",
                "args": [1, 12],
                "kwargs": {}
            },
            "min_samples_split": {
                "type": "int",
                "args": [2, 5],
                "kwargs": {}
            },
            "min_samples_leaf": {
                "type": "int",
                "args": [1, 5],
                "kwargs": {}
            },
            "max_features": {
                "type": "float",
                "args": [0.1, 1],
                "kwargs": {"log": False}
            },
            "min_impurity_decrease": {
                "type": "float",
                "args": [0, 0.2],
                "kwargs": {"log": False}
            },
            "ccp_alpha": {
                "type": "float",
                "args": [0, 0.3],
                "kwargs": {"log": False}
            },
            "base_params":{"random_state": 1}
        }

#  $$$$$$\   $$$$$$\  $$$$$$$\
# $$  __$$\ $$  __$$\ $$  __$$\
# $$ /  \__|$$ /  \__|$$ |  $$ |
# \$$$$$$\  $$ |$$$$\ $$ |  $$ |
#  \____$$\ $$ |\_$$ |$$ |  $$ |
# $$\   $$ |$$ |  $$ |$$ |  $$ |
# \$$$$$$  |\$$$$$$  |$$$$$$$  |
#  \______/  \______/ \_______/

        self.sgd_params_optim = {
            "alpha": {
                "type": "float",
                "args": [0, 1000],
                "kwargs": {"log": False}
            },
            "penalty": {
                "type": "categorical",
                "args": [['l2', 'l1', 'elasticnet']],
                "kwargs": {}
            },
            "l1_ratio": {
                "type": "float",
                "args": [0, 1],
                "kwargs": {"log": False}
            },
            "epsilon": {
                "type": "float",
                "args": [0, 1000],
                "kwargs": {"log": False}
            },
            "eta0": {
                "type": "float",
                "args": [0, 0.1],
                "kwargs": {"log": False}
            },
            "power_t": {
                "type": "float",
                "args": [-1, 1],
                "kwargs": {"log": False}
            },
            "validation_fraction": {
                "type": "float",
                "args": [0.1, 1],
                "kwargs": {"log": False}
            },
            "n_iter_no_change": {
                "type": "int",
                "args": [1, 10],
                "kwargs": {}
            },
            "base_params":{"max_iter": 150000}
        }

#  $$$$$$\  $$\    $$\ $$$$$$$\
# $$  __$$\ $$ |   $$ |$$  __$$\
# $$ /  \__|$$ |   $$ |$$ |  $$ |
# \$$$$$$\  \$$\  $$  |$$$$$$$  |
#  \____$$\  \$$\$$  / $$  __$$<
# $$\   $$ |  \$$$  /  $$ |  $$ |
# \$$$$$$  |   \$  /   $$ |  $$ |
#  \______/     \_/    \__|  \__|

        self.svr_params_optim = {
            "kernel": {
                "type": "categorical",
                "args": [['linear', 'poly', 'rbf']],
                "kwargs": {}
            },
            "degree": {
                "type": "int",
                "args": [2, 3],
                "kwargs": {}
            },
            "C": {
                "type": "float",
                "args": [0.01, 1000],
                "kwargs": {"log": False}
            },
            "coef0": {
                "type": "float",
                "args": [0, 1],
                "kwargs": {"log": False}
            },
            "epsilon": {
                "type": "float",
                "args": [0.1, 1000],
                "kwargs": {"log": False}
            },
            "gamma": {
                "type": "float",
                "args": [0.005, 0.5],
                "kwargs": {"log": False}
            },
            "base_params": {"max_iter": 150000}
        }

# $$$$$$$\   $$$$$$\  $$$$$$$\
# $$  __$$\ $$  __$$\ $$  __$$\
# $$ |  $$ |$$ /  $$ |$$ |  $$ |
# $$$$$$$  |$$$$$$$$ |$$$$$$$  |
# $$  ____/ $$  __$$ |$$  __$$<
# $$ |      $$ |  $$ |$$ |  $$ |
# $$ |      $$ |  $$ |$$ |  $$ |
# \__|      \__|  \__|\__|  \__|

        self.par_params_optim = {
            "n_iter_no_change": {
                "type": "int",
                "args": [1, 10],
                "kwargs": {}
            },
            "validation_fraction": {
                "type": "float",
                "args": [0.1, 1],
                "kwargs": {}
            },
            "C": {
                "type": "float",
                "args": [0.1, 5],
                "kwargs": {"log": False}
            },
            "epsilon": {
                "type": "float",
                "args": [0, 100],
                "kwargs": {"log": False}
            },
            "base_params":{"max_iter": 150000}
        }



    def set_param(self,
                model_params_optim: dict,
                param_name: str,
                args: List = None,
                kwargs: Dict[str, Any] = None
                ) -> Dict:

        if args is not None:
            model_params_optim[param_name]["args"] = args
        if kwargs is not None:
            model_params_optim[param_name]["kwargs"] = kwargs

        return model_params_optim
