from typing import List, Tuple, Any, Dict, Literal


class ModelsParams():
    """Класс содержащий параметры моделей для оптимизации.
    'base_params': параметры по умолчанию
    """
    def __init__(self):

# $$\   $$\ $$\   $$\ $$\   $$\
# $$ | $$  |$$$\  $$ |$$$\  $$ |
# $$ |$$  / $$$$\ $$ |$$$$\ $$ |
# $$$$$  /  $$ $$\$$ |$$ $$\$$ |
# $$  $$<   $$ \$$$$ |$$ \$$$$ |
# $$ |\$$\  $$ |\$$$ |$$ |\$$$ |
# $$ | \$$\ $$ | \$$ |$$ | \$$ |
# \__|  \__|\__|  \__|\__|  \__|

        self.knn = {
            "n_neighbors": {
                "type": "int",
                "args": [1, 10],
                "kwargs": {}
            },
            "p": {
                "type": "int",
                "args": [1, 3],
                "kwargs": {}
            },
            "weights": {
                "type": "categorical",
                "args": [['uniform', 'distance']],
                "kwargs": {}
            },
            "base_params": {}
        }

# $$\
# $$ |
# $$ |      $$$$$$\   $$$$$$$\  $$$$$$$\  $$$$$$\
# $$ |      \____$$\ $$  _____|$$  _____|$$  __$$\
# $$ |      $$$$$$$ |\$$$$$$\  \$$$$$$\  $$ /  $$ |
# $$ |     $$  __$$ | \____$$\  \____$$\ $$ |  $$ |
# $$$$$$$$\\$$$$$$$ |$$$$$$$  |$$$$$$$  |\$$$$$$  |
# \________|\_______|\_______/ \_______/  \______/

        self.lasso = {
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
        self.ridge = {
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
            "base_params": {"positive": False,
                            "random_state": 1,
                            "max_iter": 150000}
        }


# $$\   $$\           $$\
# $$ |  $$ |          $$ |
# $$ |  $$ |$$\   $$\ $$$$$$$\   $$$$$$\   $$$$$$\
# $$$$$$$$ |$$ |  $$ |$$  __$$\ $$  __$$\ $$  __$$\
# $$  __$$ |$$ |  $$ |$$ |  $$ |$$$$$$$$ |$$ |  \__|
# $$ |  $$ |$$ |  $$ |$$ |  $$ |$$   ____|$$ |
# $$ |  $$ |\$$$$$$  |$$$$$$$  |\$$$$$$$\ $$ |
# \__|  \__| \______/ \_______/  \_______|\__|

        self.huber = {
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

        self.elasticnet = {
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


#  $$$$$$\                                  $$\     $$\ $$\
# $$  __$$\                                 $$ |    \__|$$ |
# $$ /  $$ |$$\   $$\  $$$$$$\  $$$$$$$\  $$$$$$\   $$\ $$ | $$$$$$\
# $$ |  $$ |$$ |  $$ | \____$$\ $$  __$$\ \_$$  _|  $$ |$$ |$$  __$$\
# $$ |  $$ |$$ |  $$ | $$$$$$$ |$$ |  $$ |  $$ |    $$ |$$ |$$$$$$$$ |
# $$ $$\$$ |$$ |  $$ |$$  __$$ |$$ |  $$ |  $$ |$$\ $$ |$$ |$$   ____|
# \$$$$$$ / \$$$$$$  |\$$$$$$$ |$$ |  $$ |  \$$$$  |$$ |$$ |\$$$$$$$\
#  \___$$$\  \______/  \_______|\__|  \__|   \____/ \__|\__| \_______|
#      \___|

        self.quantile = {
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

#  $$$$$$\   $$$$$$\  $$$$$$$\
# $$  __$$\ $$  __$$\ $$  __$$\
# $$ /  \__|$$ /  \__|$$ |  $$ |
# \$$$$$$\  $$ |$$$$\ $$ |  $$ |
#  \____$$\ $$ |\_$$ |$$ |  $$ |
# $$\   $$ |$$ |  $$ |$$ |  $$ |
# \$$$$$$  |\$$$$$$  |$$$$$$$  |
#  \______/  \______/ \_______/

        self.sgd = {
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

        self.svr = {
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

        self.par = {
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

# $$$$$$$$\ $$\                 $$\ $$\
# \__$$  __|$$ |                \__|$$ |
#    $$ |   $$$$$$$\   $$$$$$\  $$\ $$ | $$$$$$$\  $$$$$$\  $$$$$$$\
#    $$ |   $$  __$$\ $$  __$$\ $$ |$$ |$$  _____|$$  __$$\ $$  __$$\
#    $$ |   $$ |  $$ |$$$$$$$$ |$$ |$$ |\$$$$$$\  $$$$$$$$ |$$ |  $$ |
#    $$ |   $$ |  $$ |$$   ____|$$ |$$ | \____$$\ $$   ____|$$ |  $$ |
#    $$ |   $$ |  $$ |\$$$$$$$\ $$ |$$ |$$$$$$$  |\$$$$$$$\ $$ |  $$ |
#    \__|   \__|  \__| \_______|\__|\__|\_______/  \_______|\__|  \__|

        self.theilsen = {
            "n_subsamples": {
                "type": "int",
                "args": [1, 10],
                "kwargs": {}
            },
            "max_subpopulation": {
                "type": "float",
                "args": [5, 100],
                "kwargs": {"log": False}
            },
            "base_params":{"max_iter": 150000,
                           "random_state": 1}
        }

# $$$$$$$\                                          $$$$$$$\  $$\       $$\
# $$  __$$\                                         $$  __$$\ \__|      $$ |
# $$ |  $$ | $$$$$$\  $$\   $$\  $$$$$$\   $$$$$$$\ $$ |  $$ |$$\  $$$$$$$ | $$$$$$\   $$$$$$\
# $$$$$$$\ | \____$$\ $$ |  $$ |$$  __$$\ $$  _____|$$$$$$$  |$$ |$$  __$$ |$$  __$$\ $$  __$$\
# $$  __$$\  $$$$$$$ |$$ |  $$ |$$$$$$$$ |\$$$$$$\  $$  __$$< $$ |$$ /  $$ |$$ /  $$ |$$$$$$$$ |
# $$ |  $$ |$$  __$$ |$$ |  $$ |$$   ____| \____$$\ $$ |  $$ |$$ |$$ |  $$ |$$ |  $$ |$$   ____|
# $$$$$$$  |\$$$$$$$ |\$$$$$$$ |\$$$$$$$\ $$$$$$$  |$$ |  $$ |$$ |\$$$$$$$ |\$$$$$$$ |\$$$$$$$\
# \_______/  \_______| \____$$ | \_______|\_______/ \__|  \__|\__| \_______| \____$$ | \_______|
#                     $$\   $$ |                                            $$\   $$ |
#                     \$$$$$$  |                                            \$$$$$$  |
#                      \______/                                              \______/

        self.bayesridge = {
            "alpha_1": {
                "type": "float",
                "args": [1e-7, 1000],
                "kwargs": {"log": False}
            },
            "alpha_2": {
                "type": "float",
                "args": [1e-7, 1000],
                "kwargs": {"log": False}
            },
            "lambda_1": {
                "type": "float",
                "args": [1e-7, 1000],
                "kwargs": {"log": False}
            },
            "lambda_2": {
                "type": "float",
                "args": [1e-7, 1000],
                "kwargs": {"log": False}
            },
            "base_params":{"max_iter": 150000}
        }

#  $$$$$$\  $$$$$$$\  $$$$$$$\
# $$  __$$\ $$  __$$\ $$  __$$\
# $$ /  $$ |$$ |  $$ |$$ |  $$ |
# $$$$$$$$ |$$$$$$$  |$$ |  $$ |
# $$  __$$ |$$  __$$< $$ |  $$ |
# $$ |  $$ |$$ |  $$ |$$ |  $$ |
# $$ |  $$ |$$ |  $$ |$$$$$$$  |
# \__|  \__|\__|  \__|\_______/

        self.ard = {
            "alpha_1": {
                "type": "float",
                "args": [1e-7, 1000],
                "kwargs": {"log": False}
            },
            "alpha_2": {
                "type": "float",
                "args": [1e-7, 1000],
                "kwargs": {"log": False}
            },
            "lambda_1": {
                "type": "float",
                "args": [1e-7, 1000],
                "kwargs": {"log": False}
            },
            "lambda_2": {
                "type": "float",
                "args": [1e-7, 1000],
                "kwargs": {"log": False}
            },
            "threshold_lambda": {
                "type": "float",
                "args": [1, 10000],
                "kwargs": {"log": False}
            },
            "base_params":{"max_iter": 150000}
        }

# $$$$$$$$\                                         $$\ $$\
# \__$$  __|                                        $$ |\__|
#    $$ |   $$\  $$\  $$\  $$$$$$\   $$$$$$\   $$$$$$$ |$$\  $$$$$$\
#    $$ |   $$ | $$ | $$ |$$  __$$\ $$  __$$\ $$  __$$ |$$ |$$  __$$\
#    $$ |   $$ | $$ | $$ |$$$$$$$$ |$$$$$$$$ |$$ /  $$ |$$ |$$$$$$$$ |
#    $$ |   $$ | $$ | $$ |$$   ____|$$   ____|$$ |  $$ |$$ |$$   ____|
#    $$ |   \$$$$$\$$$$  |\$$$$$$$\ \$$$$$$$\ \$$$$$$$ |$$ |\$$$$$$$\
#    \__|    \_____\____/  \_______| \_______| \_______|\__| \_______|

        self.tweedie = {
            "power": {
                "type": "float",
                "args": [0, 3],
                "kwargs": {}
            },
            "alpha": {
                "type": "float",
                "args": [0, 1000],
                "kwargs": {}
            },
            "base_params": {"max_iter": 150000}
        }

#  $$$$$$\  $$\    $$\ $$$$$$$\
# $$  __$$\ $$ |   $$ |$$  __$$\
# $$ /  \__|$$ |   $$ |$$ |  $$ |
# \$$$$$$\  \$$\  $$  |$$$$$$$  |
#  \____$$\  \$$\$$  / $$  __$$<
# $$\   $$ |  \$$$  /  $$ |  $$ |
# \$$$$$$  |   \$  /   $$ |  $$ |
#  \______/     \_/    \__|  \__|

        self.svr = {
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

# $$$$$$$\  $$$$$$$$\
# $$  __$$\ \__$$  __|
# $$ |  $$ |   $$ |    $$$$$$\   $$$$$$\   $$$$$$\
# $$ |  $$ |   $$ |   $$  __$$\ $$  __$$\ $$  __$$\
# $$ |  $$ |   $$ |   $$ |  \__|$$$$$$$$ |$$$$$$$$ |
# $$ |  $$ |   $$ |   $$ |      $$   ____|$$   ____|
# $$$$$$$  |   $$ |   $$ |      \$$$$$$$\ \$$$$$$$\
# \_______/    \__|   \__|       \_______| \_______|

        self.dt = {
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

# $$$$$$$\  $$$$$$$$\                                           $$\
# $$  __$$\ $$  _____|                                          $$ |
# $$ |  $$ |$$ |       $$$$$$\   $$$$$$\   $$$$$$\   $$$$$$$\ $$$$$$\
# $$$$$$$  |$$$$$\    $$  __$$\ $$  __$$\ $$  __$$\ $$  _____|\_$$  _|
# $$  __$$< $$  __|   $$ /  $$ |$$ |  \__|$$$$$$$$ |\$$$$$$\    $$ |
# $$ |  $$ |$$ |      $$ |  $$ |$$ |      $$   ____| \____$$\   $$ |$$\
# $$ |  $$ |$$ |      \$$$$$$  |$$ |      \$$$$$$$\ $$$$$$$  |  \$$$$  |
# \__|  \__|\__|       \______/ \__|       \_______|\_______/    \____/

        self.rf = {
            "criterion": {
                "type": "categorical",
                "args": [['squared_error', 'friedman_mse', 'absolute_error', 'poisson']],
                "kwargs": {}
            },
            "max_depth": {
                "type": "int",
                "args": [1, 10],
                "kwargs": {}
            },
            "n_estimators": {
                "type": "int",
                "args": [50, 150],
                "kwargs": {}
            },
            "max_leaf_nodes": {
                "type": "int",
                "args": [5, 15],
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
                "type": "categorical",
                "args": [['sqrt', 'log2', 1]],
                "kwargs": {}
            },
            "min_impurity_decrease": {
                "type": "float",
                "args": [0, 0.2],
                "kwargs": {"log": False}
            },
            "min_weight_fraction_leaf": {
                "type": "float",
                "args": [0, 0.5],
                "kwargs": {"log": False}
            },
            "ccp_alpha": {
                "type": "float",
                "args": [0, 0.3],
                "kwargs": {"log": False}
            },
            "base_params":{"random_state": 1}
        }

# $$$$$$$$\           $$$$$$$$\
# $$  _____|          \__$$  __|
# $$ |      $$\   $$\    $$ |    $$$$$$\   $$$$$$\   $$$$$$\   $$$$$$$\
# $$$$$\    \$$\ $$  |   $$ |   $$  __$$\ $$  __$$\ $$  __$$\ $$  _____|
# $$  __|    \$$$$  /    $$ |   $$ |  \__|$$$$$$$$ |$$$$$$$$ |\$$$$$$\
# $$ |       $$  $$<     $$ |   $$ |      $$   ____|$$   ____| \____$$\
# $$$$$$$$\ $$  /\$$\    $$ |   $$ |      \$$$$$$$\ \$$$$$$$\ $$$$$$$  |
# \________|\__/  \__|   \__|   \__|       \_______| \_______|\_______/

        self.extr = {
            "criterion": {
                "type": "categorical",
                "args": [['squared_error', 'friedman_mse', 'absolute_error', 'poisson']],
                "kwargs": {}
            },
            "max_depth": {
                "type": "int",
                "args": [1, 10],
                "kwargs": {}
            },
            "n_estimators": {
                "type": "int",
                "args": [50, 150],
                "kwargs": {}
            },
            "max_leaf_nodes": {
                "type": "int",
                "args": [5, 15],
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
                "type": "categorical",
                "args": [['sqrt', 'log2', 1]],
                "kwargs": {}
            },
            "min_impurity_decrease": {
                "type": "float",
                "args": [0, 0.2],
                "kwargs": {"log": False}
            },
            "min_weight_fraction_leaf": {
                "type": "float",
                "args": [0, 0.5],
                "kwargs": {"log": False}
            },
            "ccp_alpha": {
                "type": "float",
                "args": [0, 0.3],
                "kwargs": {"log": False}
            },
            "base_params":{"random_state": 1}
        }

# $$\   $$\  $$$$$$\  $$$$$$$\
# $$ |  $$ |$$  __$$\ $$  __$$\
# \$$\ $$  |$$ /  \__|$$ |  $$ |
#  \$$$$  / $$ |$$$$\ $$$$$$$\ |
#  $$  $$<  $$ |\_$$ |$$  __$$\
# $$  /\$$\ $$ |  $$ |$$ |  $$ |
# $$ /  $$ |\$$$$$$  |$$$$$$$  |
# \__|  \__| \______/ \_______/

        self.xgb = {
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

# $$\        $$$$$$\  $$$$$$$\  $$\      $$\
# $$ |      $$  __$$\ $$  __$$\ $$$\    $$$ |
# $$ |      $$ /  \__|$$ |  $$ |$$$$\  $$$$ |
# $$ |      $$ |$$$$\ $$$$$$$\ |$$\$$\$$ $$ |
# $$ |      $$ |\_$$ |$$  __$$\ $$ \$$$  $$ |
# $$ |      $$ |  $$ |$$ |  $$ |$$ |\$  /$$ |
# $$$$$$$$\ \$$$$$$  |$$$$$$$  |$$ | \_/ $$ |
# \________| \______/ \_______/ \__|     \__|

        self.lgbm = {
            "learning_rate": {
                "type": "float",
                "args": [5e-4, 0.1],
                "kwargs": {"log": False}
            },
            "tree_learner": {
                "type": "categorical",
                "args": [['serial', 'feature', 'data']],
                "kwargs": {}
            },
            "max_depth": {
                "type": "int",
                "args": [1, 5],
                "kwargs": {}
            },
            "num_leaves": {
                "type": "int",
                "args": [2, 20],
                "kwargs": {}
            },
            "reg_lambda": {
                "type": "float",
                "args": [0, 100],
                "kwargs": {"log": False}
            },
            "reg_alpha": {
                "type": "float",
                "args": [0, 100],
                "kwargs": {"log": False}
            },
            "min_child_samples": {
                "type": "int",
                "args": [1, 20],
                "kwargs": {}
            },
            "feature_fraction": {
                "type": "float",
                "args": [0.1, 1],
                "kwargs": {"log": False}
            },
            "min_data_in_leaf": {
                "type": "int",
                "args": [2, 20],
                "kwargs": {}
            },
            "n_estimators": {
                "type": "int",
                "args": [10, 200],
                "kwargs": {}
            },
            "base_params":{"random_state": 1,
                           "verbose": -1}
        }

#  $$$$$$\    $$\                         $$\       $$\
# $$  __$$\   $$ |                        $$ |      \__|
# $$ /  \__|$$$$$$\    $$$$$$\   $$$$$$$\ $$ |  $$\ $$\ $$$$$$$\   $$$$$$\
# \$$$$$$\  \_$$  _|   \____$$\ $$  _____|$$ | $$  |$$ |$$  __$$\ $$  __$$\
#  \____$$\   $$ |     $$$$$$$ |$$ /      $$$$$$  / $$ |$$ |  $$ |$$ /  $$ |
# $$\   $$ |  $$ |$$\ $$  __$$ |$$ |      $$  _$$<  $$ |$$ |  $$ |$$ |  $$ |
# \$$$$$$  |  \$$$$  |\$$$$$$$ |\$$$$$$$\ $$ | \$$\ $$ |$$ |  $$ |\$$$$$$$ |
#  \______/    \____/  \_______| \_______|\__|  \__|\__|\__|  \__| \____$$ |
#                                                                 $$\   $$ |
#                                                                 \$$$$$$  |
#                                                                  \______/

        self.stack = {
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
            "max_iter": {
                "type": "categorical",
                "args": [[150000]],
                "kwargs": {}
            },
            "base_params": {}
        }
#                           $$\     $$\                       $$\
#                           $$ |    $$ |                      $$ |
# $$$$$$\$$$$\   $$$$$$\  $$$$$$\   $$$$$$$\   $$$$$$\   $$$$$$$ | $$$$$$$\
# $$  _$$  _$$\ $$  __$$\ \_$$  _|  $$  __$$\ $$  __$$\ $$  __$$ |$$  _____|
# $$ / $$ / $$ |$$$$$$$$ |  $$ |    $$ |  $$ |$$ /  $$ |$$ /  $$ |\$$$$$$\
# $$ | $$ | $$ |$$   ____|  $$ |$$\ $$ |  $$ |$$ |  $$ |$$ |  $$ | \____$$\
# $$ | $$ | $$ |\$$$$$$$\   \$$$$  |$$ |  $$ |\$$$$$$  |\$$$$$$$ |$$$$$$$  |
# \__| \__| \__| \_______|   \____/ \__|  \__| \______/  \_______|\_______/

    def set_param(self,
                model_params: dict,
                param_name: str,
                args: List = None,
                kwargs: Dict[str, Any] = None
                ) -> Dict:
        """Изменяет указанный параметр модели

        Args:
            model_params (dict): Параметры модели
            param_name (str): Название параметра для измененеия
            args (List, optional): Измененные значения параметров. Defaults to None.
            kwargs (Dict[str, Any], optional): Словарь с параметрами (дополнительные параметры или базовые). Defaults to None.

        Returns:
            Dict: Словарь с измененными параметрами
        """
        if param_name == "base_params":
            model_params[param_name] = kwargs
        elif args is not None:
            model_params[param_name]["args"] = args
        elif kwargs is not None:
            model_params[param_name]["kwargs"] = kwargs

        return model_params
