from typing import Tuple, Literal, Union
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import dtreeviz
import pmdarima
from pmdarima import auto_arima
from sklearn.base import clone


def get_feature_importance_df(X: pd.DataFrame,
                              model: BaseEstimator) -> pd.DataFrame:
    """Формирует датафрейм с важностью признаков для обученной модели.

    Args:
        X (pd.DataFrame): Исходный датафрейм
        model (BaseEstimator): Обученная модель

    Returns:
        pd.DataFrame: Датафрейм с двумя столбцами:
         - 'features' (название признака)
         - 'importances' (вес признака или показатель важности)
    """
    plt.style.use('ggplot')

    try:
        # Если у модели есть .coef_ (линейные модели)
        sorted_inds = np.abs(model.coef_).argsort()[::-1]
        colnames = X.columns[sorted_inds]
        coefs = model.coef_[sorted_inds]
        feature_importances_df = pd.DataFrame({
            'features': colnames,
            'importances': coefs.round(3)
        })
    except AttributeError:
        # Если у модели есть .feature_importances_ (деревья, ансамбли)
        importances = model.feature_importances_
        feature_importances_df = (
            pd.DataFrame({
                'features': X.columns,
                'importances': importances
            })
            .sort_values(by='importances', ascending=False)
            .reset_index(drop=True)
        )
    feature_importances_df['importances_rel'] = round(feature_importances_df['importances'] / abs(feature_importances_df['importances']).sum(), 2)

    return feature_importances_df

def get_importances_barplot(X: pd.DataFrame,
                            model: BaseEstimator,
                            model_name: str="model",
                            figsize: Tuple[int] = (20,10),
                            save_dir: str = None,
                            show_shap: bool = True,
                            y: pd.Series = None #Только для визуализации дерева
                            ) -> pd.DataFrame:
    """Рисует столбчатый график важности факторов

    Args:
        X (pd.DataFrame): Исходный датафрейм
        model (BaseEstimator): Обученная модель
        model_name (str, optional): Название модели для отображения на графике. Defaults to "model".
        figsize (Tuple[int], optional): Размер графика. Defaults to (20,10).
        save_dir (str, optional): Директория для сохранения графика. Defaults to None.
        show_shap (bool, optional): SHAP-важность факторов. Defaults to True.
        y (pd.Series, optional): Целевые переменные для визуализации дерева решений. Defaults to None#Толькодлявизуализациидерева.

    Returns:
        pd.DataFrame: Датафрейм важности признаков
    """
    plt.style.use('ggplot')

    #---------------------------------------------------------------------------#
    # Важность признаков для моделей с аттрибутами .coef_ / .feature_importances_
    #---------------------------------------------------------------------------#
    feature_importances_df=None
    try:
        feature_importances_df = get_feature_importance_df(X=X, model=model)


        figure, ax = plt.subplots(1,1, figsize=figsize)
        ax = sns.barplot(data=feature_importances_df,
                        y='features',
                        x='importances_rel',
                        ax=ax, orient='h',
                        color='r',
                        edgecolor='black',
                        width=0.3)

        for p in ax.patches:
            ax.annotate(format(p.get_width(), '.2f'),  # Значение на конце столбца
                    (p.get_width(), p.get_y() + p.get_height() / 2),  # Позиция текста
                    ha='left', va='center',
                    xytext=(2, 0),  # Смещение текста вправо
                    textcoords='offset points')

        ax.set_title(f'Feature importances. {model_name}')

        if save_dir is not None:
            os.makedirs(name=save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/FE_bar_{model_name}.png", dpi=300)
        plt.show()
    except Exception as e:

        print(e)

    #------------------------------#
    # Важность признаков SHAP
    #------------------------------#
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)


    if save_dir is not None:
        shap.summary_plot(shap_values, show=False)
        plt.savefig(f"{save_dir}/SHAP_summary_{model_name}.jpg", bbox_inches='tight')
        plt.close()

        shap.plots.bar(shap_values, show=False, max_display=None)
        plt.savefig(f"{save_dir}/SHAP_bar_{model_name}.jpg", bbox_inches='tight')
        plt.close()

        #------------------------------#
        # Визуализация дерева решений
        #------------------------------#
        if y is not None:
            try:
                dtreeviz.model(model=model,
                       X_train=X,
                       y_train=y,
                       feature_names=X.columns,
                       target_name='y'
                       ).view(scale=3, ticks_fontsize=5, label_fontsize=5).save(f'{save_dir}/DT_viz.svg')
            except Exception:
                ...
    if show_shap:
        shap.summary_plot(shap_values, show=True)
        plt.close()
        shap.plots.bar(shap_values, show=True)
    if get_importances_barplot is not None:
        return feature_importances_df

def get_feature_contrib(
    X_orig: pd.DataFrame,
    model: BaseEstimator,
    scaler: StandardScaler,
    show_plot: bool=True,
    title: str = "Вклад факторов в предсказание"
) -> pd.DataFrame:
    """Расчёт вкладов факторов для линейной модели,
    обученной на стандартизированных данных

    Args:
        X_orig (pd.DataFrame): Датафрейм с исходными факторами
        model (BaseEstimator): Обученная модель на стандартизированных данных
        scaler (StandardScaler): Скейлер, которым производилась масштабирование

    Returns:
        pd.DataFrame: Датафрейм с вкладом кажого фактора в исходном мсаштабе,
        с включенным свободным членом в каждый признак.
        В сумме вклады дают значение целевой переменной
    """
    #-------------------------------------------
    # Получение данных после стандартизации
    #-------------------------------------------
    means = scaler.mean_
    std = scaler.scale_
    W_std = model.coef_
    b_std = model.intercept_

    #-------------------------------------------
    # Преобразование коэфф. в исходный масштаб
    #-------------------------------------------
    W_orig = W_std / std
    b_orig = b_std - np.sum((W_std * means) / std)

    #-------------------------------------------
    # Включение свободного члена в признаки
    #-------------------------------------------
    b_orig_adj = b_orig / np.sum(W_orig)
    X_with_b = X_orig + b_orig_adj

    #-------------------------------------------
    # Расчёт вкладов каждого признака
    #-------------------------------------------
    df_contrib = X_with_b * W_orig

    #-------------------------------------------
    # Сумма вкладов признака = Предсказание
    #-------------------------------------------
    contrib_sum = df_contrib.sum(axis=1)
    model_preds = model.predict(scaler.transform(X_orig))
    print(f"Сумма вкладов признаков и предсказание совпадают: {np.allclose(contrib_sum, model_preds)} -> {model_preds}")

    print(f"Коэффициенты в исходном масштабе: {W_orig}")
    print(f"Свободный член в исходном масштабе: {b_orig}")

    if show_plot:
        plt.style.use('ggplot')
        plt.figure(figsize=(15,10))
        ax = sns.barplot(df_contrib, orient='h', edgecolor='black', color='red', width=0.3)

        for p in ax.patches:
            ax.annotate(format(p.get_width(), '.0f'),  # Значение на конце столбца
                        (p.get_width(), p.get_y() + p.get_height() / 2),  # Позиция текста
                        ha='left', va='center',
                        xytext=(2, 0),  # Смещение текста вправо
                        textcoords='offset points')

        plt.title(f'{title}')
        plt.ylabel('Факторы')
        plt.xlabel('Значения');

    return df_contrib

def plot_ts_pred(y_train: pd.Series,
                y_pred_train: pd. Series,
                y_test: pd.Series,
                y_pred_test: pd.Series,
                show_all: bool=False,
                confidences=None,
                title: str = "Прогноз",
                xlabel: str = "Дата",
                ylabel: str = 'Y'
                ):

    plt.style.use('ggplot')
    plt.figure(figsize=(25, 20))
    plt.plot(y_train, label='Тренировочные данные', marker='.', color='blue')
    if y_test is not None:
        plt.plot(y_test, label='Действительные данные', color='green', marker='.')
    plt.plot(y_pred_test, label='Прогноз', color='red', linestyle='--', marker='.')

    if show_all:
        plt.plot(y_pred_train, label='Прогноз тренировочных данных', color='orange', linestyle='--', marker='.')
    if confidences is not None:
        plt.fill_between(
            y_pred_test.index,
            confidences[:, 0],
            confidences[:, 1],
            color='red',
            alpha=0.2,
            label='Доверительный интервал',
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def relative_error(y_pred: np.array,
            y_true: pd.Series,
            re_threshold: int=30) -> int:
    """Вычисляет долю объектов с относительной ошибкой меньше порогового значения

    Args:
        y_pred (np.array): Вектор предсказанных целевых переменных
        y_true (pd.Series): Вектор действительных целевых переменных
        re_threshold (int, optional): Пороовое значение относительной ошибки. Defaults to 30.

    Returns:
        int: Доля объектов, для которых значения относительной ошибки меньше порогового значения
    """

    re = abs(y_pred - y_true)
    re_relative = (re / (y_true + 1e-8)) * 100
    count_less_thresh = re_relative[re_relative < re_threshold].shape[0]
    count_less_thresh_ratio = count_less_thresh / y_true.shape[0]

    return count_less_thresh_ratio

def get_prediction(
    X: pd.DataFrame,
    y: pd.Series,
    model: BaseEstimator,
    re_threshold: int = 30,
    data_name: str = 'data_name',
    metrics_type: str = None,
    sarimax_train: bool = False,
    sarimax_forecast: int = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Возвращает датафрейм с метриками и массив предсказаний модели.

    Args:
        X (pd.DataFrame): Датафрейм для предсказания
        y (pd.Series): Целевая переменная
        model (BaseEstimator): Обученная модель
        re_threshold (int, optional): Пороовое значение относительной ошибки. Defaults to 30.
        data_name (str): Название данных. Defaults to 'data_name'
        metrics_type (str): Тип разделения данных (train / test ...) Defaults to None.
        sarimax_train(bool): Рассчет метрик для модели auto_arima (Обучающие данные)
        sarimax_forecast(int): Рассчет метрик для модели auto_arima (Период предсказания)

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Кортеж датафрейма с метриками и массив предсказаний модели
    """

    if sarimax_train:
        y_pred, confidences = model.predict_in_sample(
            start=y.index[0],
            X=X,
            return_conf_int=True)
    if sarimax_forecast is not None:
        y_pred, confidences = model.predict(
            n_periods=sarimax_forecast,
            X=X,
            return_conf_int=True)
    if sarimax_train is False and sarimax_forecast is None:
        y_pred_arr = model.predict(X)
        y_pred = pd.Series(data=y_pred_arr, index=X.index)

    RMSE = root_mean_squared_error(y, y_pred)
    MAE = mean_absolute_error(y, y_pred)
    RE = relative_error(y_pred=y_pred,
                        y_true=y,
                        re_threshold=re_threshold)
    negative = (y_pred < 0).sum()

    metrics_dict = {
        "model": model,
        "negative": negative,
        "RE": RE,
        "MAE": MAE,
        "RMSE": RMSE
    }

    # Проверяем, что в выборке больше 1 точки (иначе r2_score не посчитать)
    if y_pred.shape[0] > 1:
        R2 = r2_score(y, y_pred)
        # Чтобы избежать деления на ноль:
        denom = (y.max() - y.min())
        if denom == 0:
            NRMSE = np.nan
        else:
            NRMSE = RMSE / denom
        metrics_dict["NRMSE"] = NRMSE
        metrics_dict["R2"] = R2

    metrics_df = pd.DataFrame(metrics_dict, index=[0])

    if metrics_type is not None:
        for col_name in metrics_df.columns:
            metrics_df = metrics_df.rename(columns={col_name: f'{col_name}_{metrics_type}'})
    metrics_df['data'] = data_name

    return metrics_df, y_pred


def train_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model: BaseEstimator,
    cv_type: Literal['kf', 'loo', 'stratify', 'ts', 'group_kf'],
    metric_best: Literal['R2_val', 'RMSE_val', 'NRMSE_val', 'MAE_val', 'RE_val' ],
    stratify: Union[str, None] = 'quantile',  # только для 'stratify'
    n_splits: int = 5,                       # KFold / StratifiedKFold
    shuffle: bool = False,                   # KFold / StratifiedKFold
    train_size: int = 48,                    # TimeSeriesSplit
    val_size: int = 12,                      # TimeSeriesSplit
    data_name: Union[str, None] = None,
    groups: Union[pd.Series, np.ndarray, None] = None
) -> Tuple[BaseEstimator, dict]:
    """
    Единая функция для обучения модели с разными схемами кросс-валидации:
      - 'stratify' : StratifiedKFold (доп. стратификация по квантилям или столбцу)
      - 'loo'      : LeaveOneOut
      - 'kf'       : KFold
      - 'ts'       : TimeSeriesSplit

    Параметры
    ---------
    X : pd.DataFrame
        Матрица признаков
    y : pd.Series
        Целевые значения
    model : BaseEstimator
        Модель (sklearn), у которой есть .fit/.predict
    cv_type : {'stratify', 'loo', 'kf', 'ts'}
        Какую схему кросс-валидации использовать
    stratify : str или None
        - 'quantile': стратификация по квартилям y (4 корзины)
        - иначе: название столбца X для стратификации
    n_splits : int
        Количество сплитов (KFold / StratifiedKFold)
    shuffle : bool
        Перемешивать ли данные в KFold / StratifiedKFold
    train_size : int
        Размер обучающей выборки для TimeSeriesSplit
    val_size : int
        Размер валидации (окна) для TimeSeriesSplit
    data_name : str или None
        Название набора данных (для логов)

    Возвращает
    ---------
    best_model : BaseEstimator
        Обученная модель (либо «лучшая» — в Stratified/KFold/LOO,
                          либо последняя — в TS)
    final_result : dict
        Словарь со статистикой метрик и сами метрики по сплитам
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Словарь для хранения всех метрик train/val (включая "negative")
    results_dict = {
        'R2_train': [],
        'RMSE_train': [],
        'NRMSE_train': [],
        'MAE_train': [],
        'RE_train': [],
        'negative_train': [],

        'R2_val': [],
        'RMSE_val': [],
        'NRMSE_val': [],
        'MAE_val': [],
        'RE_val': [],
        'negative_val': []
    }
    models_history = []
    y_val_full = []
    y_pred_full = []

    # -------------------------- #
    # 1. Определяем способ split
    # -------------------------- #
    if cv_type == 'stratify':
        # Подготовка страты
        quant_25 = y.quantile(0.25)
        quant_50 = y.quantile(0.50)
        quant_75 = y.quantile(0.75)

        if stratify == 'quantile':
            def _strat_func(val: float) -> int:
                if val <= quant_25:
                    return 1
                elif val <= quant_50:
                    return 2
                elif val <= quant_75:
                    return 3
                else:
                    return 4
            strat_vector = y.apply(_strat_func)
        else:
            strat_vector = X[stratify]

        cv_splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
        )
        split_iter = cv_splitter.split(X, strat_vector)
        desc_text = f"StratifiedKFold (n_splits={n_splits})"

    elif cv_type == 'loo':
        cv_splitter = LeaveOneOut()
        split_iter = cv_splitter.split(X)
        desc_text = "LeaveOneOut"

    elif cv_type == 'kf':
        cv_splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
        )
        split_iter = cv_splitter.split(X)
        desc_text = f"KFold (n_splits={n_splits})"

    elif cv_type == 'ts':
        n_splits_ts = (len(X) - train_size) // val_size
        cv_splitter = TimeSeriesSplit(
            n_splits=n_splits_ts,
            test_size=val_size
        )
        split_iter = cv_splitter.split(X)
        desc_text = f"TimeSeriesSplit (n_splits={n_splits_ts})"
    elif cv_type == 'group_kf':
        if groups is None:
            raise ValueError("Для 'group_kf' необходимо передать параметр 'groups' (например, object id).")
        if isinstance(groups, pd.Series):
            groups_used = groups.reset_index(drop=True)
        else:
            groups_used = pd.Series(groups).reset_index(drop=True)
        cv_splitter = GroupKFold(n_splits=n_splits)
        split_iter = cv_splitter.split(X, y, groups=groups_used)
        desc_text = f"GroupKFold (n_splits={n_splits})"
    else:
        raise ValueError(f"Неверный cv_type='{cv_type}'. Допустимые: 'stratify', 'loo', 'kf', 'ts', 'group_kf'.")

    # -------------------------- #
    # 2. Основной цикл
    # -------------------------- #
    progrbar = tqdm(split_iter, total=cv_splitter.get_n_splits(X), desc=desc_text)
    for train_idx, val_idx in progrbar:
        model_i = clone(model)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_i.fit(X_train, y_train)

        # Получаем метрики на train
        metrics_train_df, y_train_pred = get_prediction(X_train, y_train, model_i)
        # Получаем метрики на val
        metrics_val_df, y_val_pred = get_prediction(X_val, y_val, model_i)

        R2_train = metrics_train_df.get("R2", np.nan)
        RMSE_train = metrics_train_df["RMSE"]
        NRMSE_train = metrics_train_df.get("NRMSE", np.nan)
        MAE_train = metrics_train_df["MAE"]
        RE_train = metrics_train_df["RE"]
        negative_train = metrics_train_df["negative"]

        R2_val = metrics_val_df.get("R2", np.nan)
        RMSE_val = metrics_val_df["RMSE"]
        NRMSE_val = metrics_val_df.get("NRMSE", np.nan)
        MAE_val = metrics_val_df["MAE"]
        RE_val = metrics_val_df["RE"]
        negative_val = metrics_val_df["negative"]

        # Заполняем общий словарь
        results_dict['R2_train'].append(R2_train)
        results_dict['RMSE_train'].append(RMSE_train)
        results_dict['NRMSE_train'].append(NRMSE_train)
        results_dict['MAE_train'].append(MAE_train)
        results_dict['RE_train'].append(RE_train)
        results_dict['negative_train'].append(negative_train)

        results_dict['R2_val'].append(R2_val)
        results_dict['RMSE_val'].append(RMSE_val)
        results_dict['NRMSE_val'].append(NRMSE_val)
        results_dict['MAE_val'].append(MAE_val)
        results_dict['RE_val'].append(RE_val)
        results_dict['negative_val'].append(negative_val)

        models_history.append(model_i)
        y_val_full.append(y_val)
        y_pred_full.append(y_val_pred)

    # -------------------------- #
    # 3. Преобразуем и ищем «лучшую» модель
    # -------------------------- #
    for k, arr in results_dict.items():
        results_dict[k] = np.array(arr)

    if cv_type in ('ts'):
        # Для 'ts' — последняя
        best_model_split = models_history[-1]

    if cv_type == "loo":
        # Логика выбора "лучшей": к примеру, минимизируем MAE_val
        if metric_best in ('RMSE_val', 'MAE_val', 'RE_val' ):
            if metric_best in ('RMSE_val', 'MAE_val'):
                best_score_ind = results_dict[metric_best].argmin()
                best_model_split = models_history[best_score_ind]
            else:
                best_score_ind = results_dict[metric_best].argmax()
                best_model_split = models_history[best_score_ind]
        else:
            raise ValueError("""Для LOO метрика выбора лучшей модели (metric_best)
                должна быть: RMSE_val, MAE_val, RE_val""")
    else:
        if metric_best in ('RMSE_val', 'MAE_val'):
            best_score_ind = results_dict[metric_best].argmin()
            best_model_split = models_history[best_score_ind]
        else:
            best_score_ind = results_dict[metric_best].argmax()
            best_model_split = models_history[best_score_ind]

    # Обучение на всех обучающих данных
    best_model = best_model_split.fit(X, y)
    # -------------------------- #
    # 4. Сводный словарь с результатами
    # -------------------------- #
    final_result = {
        'model': str(best_model),
        'data_name': data_name
    }

    for key, values in results_dict.items():
        final_result[f'{key}_macro'] = np.nanmean(values).round(3)
        final_result[f'{key}_std'] = np.nanstd(values).round(3)
        final_result[f'{key}_splits'] = np.round(values, 3)

    y_val_concat = pd.concat(y_val_full).values
    y_pred_concat = pd.concat(y_pred_full).values
    if len(y_val_concat) > 1:
        final_result['R2_val_micro'] = r2_score(y_val_concat, y_pred_concat)
        final_result['RMSE_val_micro'] = root_mean_squared_error(
            y_val_concat, y_pred_concat
        )
        final_result['MAE_val_micro'] = mean_absolute_error(
            y_val_concat, y_pred_concat
        )
        final_result['RE_val_micro'] = relative_error(y_pred=y_pred_concat,
                               y_true=y_val_concat)
        final_result['Negative_micro'] = (y_pred_concat < 0).sum()
        val_denom = (y_val_concat.max() - y_val_concat.min())
        if val_denom == 0:
            final_result['NRMSE_val_micro'] = np.nan
        else:
            final_result['NRMSE_val_micro'] = (
                final_result['RMSE_val_micro'] / val_denom
            )
    else:
        final_result['R2_val_micro'] = np.nan
        final_result['RMSE_val_micro'] = np.nan
        final_result['MAE_val_micro'] = np.nan
        final_result['RE_val_micro'] = np.nan
        final_result['NRMSE_val_micro'] = np.nan

    # Метрики разности валидационных значений и обучающих
    if cv_type == 'loo':
        if final_result['R2_train_macro'] > 0 and final_result['R2_val_micro'] > 0:
            R2_diff = abs(final_result['R2_train_macro'] - final_result['R2_val_micro'])
        else:
            R2_diff = np.inf
        RMSE_diff = abs(final_result['RMSE_train_macro'] - final_result['RMSE_val_micro'])
        MAE_diff = abs(final_result['MAE_train_macro'] - final_result['MAE_val_macro'])
    else:
        if (final_result['R2_train_splits'] - final_result['R2_val_splits']).mean() > 0:
            R2_diff = (final_result['R2_train_splits'] - final_result['R2_val_splits']).mean()
        else:
            R2_diff = np.inf
        RMSE_diff = abs(final_result['RMSE_train_splits'] - final_result['RMSE_val_splits']).mean()
        MAE_diff = abs(final_result['MAE_train_splits'] - final_result['MAE_val_splits']).mean()

    if final_result['R2_train_macro'] > 0:
        final_result['R2_diff_rel'] = R2_diff / final_result['R2_train_macro']
    else:
        final_result['R2_diff_rel'] = np.inf
    final_result['RMSE_diff_rel'] = RMSE_diff / final_result['RMSE_train_macro']
    final_result['MAE_diff_rel'] = MAE_diff / final_result['MAE_train_macro']

    clear_output()

    return best_model, final_result


def arima_train(
    # Параметры auto_arima
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    start_p=0, max_p=1,     # Диапазон значений для параметра p
    start_d=0, max_d=1,     # Диапазон значений для параметра d
    start_q=0, max_q=1,     # Диапазон значений для параметра q
    start_P=0, max_P=1,     # Диапазон значений для параметра P (сезонная AR)
    start_D=0, max_D=1,     # Диапазон значений для параметра D (сезонная I)
    start_Q=0, max_Q=1,     # Диапазон значений для параметра Q (сезонная MA)
    seasonal: bool=True,
    m=12,
    trend: Literal['c', 't', 'ct', 'n', 'ctt'] = None,
    with_intercept=True,
    trace=False,             # Показывать процесс подбора параметров
    error_action='ignore',  # Игнорировать ошибки при подборе
    suppress_warnings=True, # Подавить предупреждения
    stepwise=False,
    arima_params: dict = {},
    show_info: bool = True
    ) -> pd.DataFrame:
    """Обучение модели SARIMAX с автоматическим подбором параметров

    Args:
        X_train (pd.DataFrame): Обучающий датафрейм
        y_train (pd.Series): Обучающие целевые переменные
        X_test (pd.DataFrame): Тестовый датафрейм
        y_test (pd.Series): тестовые целевые переменные
        start_p (int, optional): Начало диапазона для параметра p (ar). Defaults to 0.
        max_p (int, optional):  Конец диапазона для параметра p (ar). Defaults to 1.
        start_d (int, optional): Начало диапазона для параметра d (i). Defaults to 0.
        max_d (int, optional): Конец диапазона для параметра d (i). Defaults to 1.
        start_q (int, optional): Начало диапазона для параметра q (ma). Defaults to 0.
        max_q (int, optional): Конец диапазона для параметра q (ma). Defaults to 1.
        start_P (int, optional): Начало диапазона для параметра P (сезонная AR). Defaults to 0.
        max_P (int, optional): Конец диапазона для параметра P (сезонная AR). Defaults to 1.
        start_D (int, optional): Начало диапазона для параметра D (сезонная I). Defaults to 0.
        max_D (int, optional): Конец диапазона для параметра D (сезонная I). Defaults to 1.
        start_Q (int, optional): Начало диапазона для параметра Q (сезонная MA). Defaults to 0.
        max_Q (int, optional): Конец диапазона для параметра Q (сезонная MA). Defaults to 1.
        m (int, optional): Периода сезонности. Defaults to 12.
        trend (Literal[&#39;c&#39;, &#39;t&#39;, &#39;ct&#39;, &#39;n&#39;, &#39;ctt&#39;], optional): Включение в модель тренда:
        - 'n': Не включать тренд
        - 'c': Константа (как  свободный член в линейной регрессии)
        - 't': Линейный тренд.
        - 'ct': Линейный тренд + константа
        - 'ctt': Квадратичный тренд
        - None: Автоматически подобрать параметр. Defaults to None.
        with_intercept (bool, optional): Включение в модель свободного члена. Defaults to True.
        trace (bool, optional): Показывать процесс подбора параметров. Defaults to False.
        arima_params (dict, optional): Дополнительные параметры модели. Defaults to {}.
        show_info (bool, optional): Показать информацию об обучении модели. Defaults to True.

    Returns:
        pd.DataFrame: Результаты обучения модели:
        - model: Обученная модель Sarimax
        - train_metrics: Метрики качества для обучающих данных
        - test_metrics: Метрики качества для тестовых данных
        - pred_df: Датафрейм с предсказаниями
    """
    #----------------------------------------------------#
    # Обучение SARIMAX с подпором параметров
    #----------------------------------------------------#
    auto_model = auto_arima(
        y=y_train,
        X=X_train,
        seasonal=seasonal,
        m=m,
        trend=trend,
        start_p=start_p, max_p=max_p,     # Диапазон значений для параметра p
        start_q=start_q, max_q=max_q,     # Диапазон значений для параметра q
        start_P=start_P, max_P=max_P,     # Диапазон значений для параметра P (сезонная AR)
        start_Q=start_Q, max_Q=max_Q,     # Диапазон значений для параметра Q (сезонная MA)
        start_d=start_d, max_d=max_d,     # Диапазон значений для параметра Q (сезонная MA)
        start_D=start_D, max_D=max_D,     # Диапазон значений для параметра Q (сезонная MA)
        trace=trace,             # Показывать процесс подбора параметров
        error_action=error_action,  # Игнорировать ошибки при подборе
        suppress_warnings=suppress_warnings, # Подавить предупреждения
        stepwise=stepwise,
        with_intercept=with_intercept,
        arima_params=arima_params
        )

    y_all = pd.concat([y_train, y_test])
    if X_train is not None:
        X_train_test_all = pd.concat([X_train, X_test])
    else:
        X_train_test_all = None
    #----------------------------------------------------#
    # Получение метрик
    #----------------------------------------------------#
    train_metrics, y_pred_train = get_prediction(
                                X=X_train,
                                y=y_train[:],
                                model=auto_model,
                                sarimax_train=True,
                                metrics_type='train')
    test_metrics, y_pred_test = get_prediction(
                                X=X_test,
                                y=y_test,
                                model=auto_model,
                                sarimax_forecast=len(y_test),
                                metrics_type='test')

    #----------------------------------------------------#
    # Создание датафрейма с предсказаниями
    #----------------------------------------------------#
    y_pred_all = pd.concat([y_pred_train, y_pred_test])
    y_pred_all.name = 'y_pred'
    y_all.name = 'y_true'

    if X_train_test_all is not None:
        pred_df = pd.concat([X_train_test_all , y_all, y_pred_all], axis=1)
    else:
        pred_df = pd.concat([ y_all, y_pred_all], axis=1)
    pred_df.drop_duplicates(inplace=True)
    pred_df['abs_error'] = pred_df['y_pred'] - pred_df['y_true']
    pred_df['rel_error'] = round(abs(pred_df['abs_error'] / pred_df['y_true']), 3) * 100
    pred_df['test_data'] = " "
    pred_df.loc[y_test.index, 'test_data'] = "X"

    #----------------------------------------------------#
    # Сохранение всех рез-ов обучения
    #----------------------------------------------------#
    train_results = {}
    train_results['model'] = auto_model
    train_results['train_metrics'] = train_metrics
    train_results['test_metrics'] = test_metrics
    train_results['pred_df'] = pred_df

    #----------------------------------------------------#
    # Визуализация результатов обучения
    #----------------------------------------------------#
    if show_info:
        print(auto_model.summary())
        diagnostic_plot = auto_model.plot_diagnostics(figsize=(10, 10))
        plot_ts_pred(
            y_train=y_train,
            y_pred_train=y_pred_train,
            y_test=y_test,
            y_pred_test=y_pred_test,
            show_all=True
        )
    return train_results

def arima_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    arima_trained_params: dict,
    show_info:bool = True,
    title: str = 'Прогноз',
    ylabel: str = 'Целевая переменная',
    show_conf: bool = True,
    ) -> pd.DataFrame:
    """Предсказание временного ряда на заданный период

    Args:
        X_train (pd.DataFrame): Обучающий датафрейм
        y_train (pd.Series): Обучающие целевые переменные
        X_pred (pd.DataFrame): Датафрейм с экзогенными переменными для периода предсказания
        arima_trained_params (dict): Параметры обученной модели SARIMAX
        show_info (bool, optional): Показывать график с предсказаниями и результаты обучения модели. Defaults to True.
        title (str, optional): Название графика. Defaults to 'Прогноз'.
        ylabel (str, optional): Название целевой переменной. Defaults to 'Y'.
        freq_tick (str, optional): Интервал дат для отображения на графике. Defaults to None.
        show_conf (bool, optional): Показывать доверительный интервал. Defaults to True.

    Returns:
        pd.DataFrame: Результаты обучения модели:
        - model: Обученная модель Sarimax
        - train_metrics: Метрики качества для обучающих данных
        - pred_df: Датафрейм с предсказаниями
    """
    auto_model = auto_arima(
        y=y_train,
        X=X_train,
        **arima_trained_params
        )

    train_metrics, y_pred_train = get_prediction(
                            X=X_train,
                            y=y_train,
                            model=auto_model,
                            sarimax_train=True,
                            metrics_type='train')

    n_periods = len(X_pred)
    y_pred, conf = auto_model.predict(
            n_periods=n_periods,
            X=X_pred,
            return_conf_int=True)

    pred_df = pd.DataFrame(data=y_pred, columns=['Прогноз'])

    if show_info:
        print(auto_model.summary())
        diagnostic_plot = auto_model.plot_diagnostics(figsize=(10, 10))
        if show_conf:
            plot_ts_pred(
                y_train=y_train,
                y_pred_train=y_pred_train,
                y_test=None,
                y_pred_test=y_pred,
                show_all=True,
                confidences=conf
            )
        else:
            plot_ts_pred(
                y_train=y_train,
                y_pred_train=y_pred_train,
                y_test=None,
                y_pred_test=y_pred,
                show_all=True,
                title=title,
                ylabel=ylabel
            )
    #----------------------------------------------------#
    # Сохранение всех рез-ов обучения
    #----------------------------------------------------#
    train_results = {}
    train_results['model'] = auto_model
    train_results['train_metrics'] = train_metrics
    train_results['pred_df'] = pred_df

    return train_results

def get_feat_importance_arima(model: pmdarima.arima.arima.ARIMA,
                              start_inx: int,
                              end_inx: int
                              ) -> pd.Series:
    """Построение графика важности признаков для SARIMAX

    Args:
        model (pmdarima.arima.arima.ARIMA): Обученная модель SARIMAX
        start_inx (int): Начальный индекс коэффициентов модели для отображения
        end_inx (int): Конечный индекс коэффициентов модели для отображения

    Returns:
        pd.Series: Коэффициенты модели
    """
    coefs = model.params()
    coefs_rel = (coefs[start_inx:end_inx] / abs(coefs[start_inx:end_inx]).sum())
    coefs_rel_sorted =  (coefs_rel).reindex((coefs_rel).abs().sort_values(ascending=True).index)
    coefs_names = coefs_rel_sorted.index

    fig = plt.figure(figsize=(15,10))
    plt.barh(coefs_names, coefs_rel_sorted, color="red", height=0.3, edgecolor='black' )

    for index, value in enumerate(coefs_rel_sorted):
        plt.text(value, index, f"{value:.2f}", va="center", ha="left" if value > 0 else "right")

    plt.title("Важность факторов")
    plt.xlabel("Относительные коэффициенты")
    plt.ylabel("Параметры модели")
    plt.show()

    return coefs
