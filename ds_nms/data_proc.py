from typing import List, Tuple, Dict, Literal, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    Normalizer,
    MinMaxScaler,
    TargetEncoder,
    PolynomialFeatures)
from sklearn.decomposition import PCA
from IPython.display import display
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_duplicated_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Возвращает датафрейм с дубликатами в указанном столбце.

    Args:
        df (pd.DataFrame): Исходный датафрейм
        column (str): Имя столбца для поиска дубликатов.

    Returns:
        pd.DataFrame: Датафрейм с дубликатами, отсортированный по указанному столбцу.
    """
    duplicate_mask = df[column].duplicated(keep=False)
    result_df = df[duplicate_mask].sort_values(by=column)

    return result_df

def features_separate(df: pd.DataFrame, threshold: int) -> Tuple[List[str], List[str]]:
    """
    Разделяет признаки датафрейма на категориальные и числовые на основе порога уникальных значений.

    Args:
        df (pd.DataFrame): Исходный датафрейм.
        threshold (int): Порог уникальных значений для классификации категориальных признаков.

    Returns:
        Tuple[List[str], List[str]]: Кортеж списков категориальных и числовых признаков.
    """
    categorical_columns = []
    numerical_columns = []

    for column_name in df.columns:
        if df[column_name].nunique() < threshold:
            categorical_columns.append(column_name)
        else:
            numerical_columns.append(column_name)

    return categorical_columns, numerical_columns

def get_stratified_df(X: pd.DataFrame, feature: pd.Series,
                        col_name: str="strat",
                        display_info: bool=True) -> pd.DataFrame:
    """Добавление в исходный датафрейм квартилей указанной переменной

    Args:
        X (pd.DataFrame): Исходный датафрейм.
        feature (pd.Series): Переменная для стратификации.
        col_name (str, optional): Имя столбца для уровней стратификации. Defaults to "strat".
        display_info (bool, optional): Отображать информацию о стратификации. Defaults to True.

    Returns:
        pd.DataFrame: Датафрейм с добавленным столбцом стратификации по квартилям.
    """
    q1 = feature.quantile(0.25)
    q2 = feature.quantile(0.50)
    q3 = feature.quantile(0.75)

    df_stratify = X.copy()
    df_stratify[col_name] = feature.apply(
        lambda x: 0 if x <= q1 else 1 if x <= q2 else 2 if x <= q3 else 3
    )

    if display_info:
        display(df_stratify.head(3))
        display(df_stratify[col_name].value_counts(normalize=True))

    return df_stratify

def df_scaling(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    numerical_columns: List[str],
    scaler: StandardScaler | MinMaxScaler | Normalizer | RobustScaler,
    return_scaler: bool = False
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame],
                            Tuple[pd.DataFrame, pd.DataFrame, Union[StandardScaler, MinMaxScaler, Normalizer, RobustScaler]]]:
    """Масштабирование числовых данных в тренировочном и тестовом наборах.

    Args:
        df_train (pd.DataFrame): Датафрейм с тренировочными данными.
        df_test (pd.DataFrame): Датафрейм с тестовыми данными.
        numerical_columns (List[str]):  Список числовых колонок для масштабирования.
        scaler (StandardScaler | MinMaxScaler | Normalizer | RobustScaler): Объект стандартизатора.
        return_scaler (bool, optional): Если True, возвращает scaler. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:  Масштабированные тренировочные и тестовые данные, опционально scaler.
    """
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

    display(df_train_num_scaled.describe().round(1))
    display(df_test_num_scaled.describe().round(1))

    if return_scaler:
        return df_train_num_scaled, df_test_num_scaled, train_scaler

    return df_train_num_scaled, df_test_num_scaled


def drop_outliers_iso(X: pd.DataFrame, y: pd.Series,
                contamination: float=0.04,
                n_estimators: int=100) -> Tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame]:
    """Удаляет выбросы из данных с использованием IsolationForest.

    Args:
        X (pd.DataFrame): Датафрейм с данными
        y (pd.Series): Целевая переменная
        contamination (float, optional): Доля выбросов в данных. Defaults to 0.04.
        n_estimators (int, optional): Количество деревьев в IsolationForest. Defaults to 100.

    Returns:
        Tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame]: X без выбросов, y без выбросов, датафрейм выбросов
    """
    irf = IsolationForest(contamination=contamination,
                    n_estimators=n_estimators,
                    random_state=1)
    irf.fit(X)
    prediction = irf.predict(X)

    clear_mask = prediction == 1
    outlier_mask = prediction == -1

    X_cleared = X[clear_mask]
    y_cleared = y[clear_mask]
    outliers = X[outlier_mask]

    print(f"Удалено {outlier_mask.sum()} объектов из {X.shape[0]}")

    return X_cleared, y_cleared, outliers


def drop_outliers_tuk(X: pd.DataFrame, y: pd.Series,
                feature : str,
                left:float=1.5, right:float=1.5,
                log_scale:bool=False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Удаляет выбросы из признаков и целевой переменной с использованием метода Тьюки

    Args:
        X (pd.DataFrame): Датафрейм с данными
        y (pd.Series): Целевая переменная
        feature (str): Имя столбца, по которому вычисляются выбросы.
        left (float, optional): Коэффициент для нижней границы. Defaults to 1.5.
        right (float, optional): Коэффициент для верхней границы. Defaults to 1.5.
        log_scale (bool, optional): Применить логарифмирование к данным перед вычислением. Defaults to False.

    Raises:
        ValueError: ошибка, если в данных есть отрицательные значения (логарифмирование невозможно)

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]: X без выбросов, y без выбросов, датафрейм выбросов
    """
    if log_scale:
        if (X[feature] <= 0).any():
            raise ValueError(f"Столбец '{feature}' содержит значения <= 0, логарифмирование невозможно.")
        x = np.log(X[feature]+1)
    else:
        x = X[feature]
    quant_25, quant_75 = x.quantile(0.25), x.quantile(0.75)
    IQR = quant_75 - quant_25
    bond_low = quant_25 - IQR * left
    bond_up = quant_75 + IQR * right

    cleaned_mask = (x >= bond_low) & (x <= bond_up)
    outlier_mask = ~cleaned_mask

    cleaned_data = X[cleaned_mask]
    cleaned_y = y[cleaned_mask]

    outliers_data = X[outlier_mask]
    outliers_y = y[outlier_mask]

    print(f"Удалено {outlier_mask.sum()} объектов из {X.shape[0]}")

    return cleaned_data, cleaned_y, outliers_data

def get_pca(X_train: pd.DataFrame, X_test: pd.DataFrame,
            columns_pca: Dict[str, List[str]],
            n_components: int=1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Преобразует указанные группы признаков с использованием PCA и добавляет результат как новые столбцы

    Args:
        X_train (pd.DataFrame): Тренировочный набор данных
        X_test (pd.DataFrame): Тестовый набор данных
        columns_pca (Dict[str, List[str]]): Словарь, где ключ - имя нового признака,
                                            значение - список столбцов для PCA.
        n_components (int, optional): Количество компонент для PCA. Defaults to 1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Кортеж преобразованных тренировочных и тестовых наборов данных
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    for column_name, columns_lst in columns_pca.items():
        pca = PCA(n_components=n_components)

        train_arr = pca.fit_transform(X_train.loc[:, columns_lst])
        test_arr = pca.transform(X_test.loc[:, columns_lst])

        X_train[column_name] = train_arr[:, 0] if n_components == 1 else train_arr
        X_test[column_name] = test_arr[:, 0] if n_components == 1 else test_arr

        X_train.drop(columns=columns_lst, inplace=True)
        X_test.drop(columns=columns_lst, inplace=True)

    return X_train, X_test

def get_VIF(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет факторы инфляции дисперсии (VIF) для каждого столбца в датафрейме.

    Args:
        df (pd.DataFrame): Датафрейм с числовыми признаками.

    Returns:
        pd.DataFrame: Датафрейм с именами столбцов и их значениями VIF.
    """

    df = df.copy()
    df["const"] = 1

    vif_data = pd.DataFrame({
        "Feature": df.columns,
        "VIF": [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    }, index=df.columns)

    # Исключаем VIF для константы
    vif_data = vif_data[vif_data["Feature"] != "const"].sort_values(by="VIF", ascending=False)

    display(vif_data[["VIF"]].sort_values(by="VIF", ascending=False).T)

    return vif_data

def plot_corrmatrix(df: pd.DataFrame,
                    target: pd.Series=None,
                    calc_det: bool = False,
                    method: str = 'pearson') -> None:
    """
    Строит тепловую карту корреляционной матрицы с дополнительной информацией (ранг и детерминант).

    Args:
        df (pd.DataFrame): Датафрейм с данными.
        target (pd.Series): Целевая переменная Defaults to None.
        calc_det (bool, optional): Вычислять ранг и детерминант корреляционной матрицы. Defaults to False.
        method (str, optional): Метод вычисления корреляции ('pearson', 'kendall', 'spearman'). Defaults to 'pearson'.

    Returns:
        None
    """

    if target is None:
        corr_matrix = df.corr(method=method)
    else:
        df = pd.concat((df, target), axis=1)
        corr_matrix = df.corr(method=method)

    fig = plt.figure(figsize=(corr_matrix.shape[0], corr_matrix.shape[1]))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, square=False)
    plt.title(f'Корреляционная матрица ({method})', fontsize=14)
    plt.show()

    if calc_det:
        if target is None:
            rank = np.linalg.matrix_rank(corr_matrix)
            determinant = np.linalg.det(corr_matrix)
            print(f'Ранг корреляционной матрицы: {rank}')
            print(f'Размер корреляционной матрицы: {corr_matrix.shape[0]}x{corr_matrix.shape[1]}')
            print(f'Детерминант корреляционной матрицы: {determinant:.3f}')
        else:
            corr_matrix = corr_matrix.iloc[:-1, :-1]
            rank = np.linalg.matrix_rank(corr_matrix)
            determinant = np.linalg.det(corr_matrix)
            print(f'Ранг корреляционной матрицы: {rank}')
            print(f'Размер корреляционной матрицы: {corr_matrix.shape[0]}x{corr_matrix.shape[1]}')
            print(f'Детерминант корреляционной матрицы: {determinant:.3f}')


def get_polyfeatures(X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    degree: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Создаёт полиномиальные признаки для тренировочного и тестового наборов.

    Args:
        X_train (pd.DataFrame): Датафрейм с тренировочными данными.
        X_test (pd.DataFrame): Датафрейм с тестовыми данными.
        degree (int, optional): Степень полинома. Defaults to 2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Тренировочный набор с полиномиальными признаками.
            - Тестовый набор с полиномиальными признаками.
    """

    col_names = list(X_train.columns)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly.fit(X_train)
    X_train_poly_arr = poly.transform(X_train)
    X_test_poly_arr = poly.transform(X_test)

    df_columns = poly.get_feature_names_out(col_names)

    X_train_poly = pd.DataFrame(X_train_poly_arr, columns=df_columns, index=X_train.index)
    X_test_poly = pd.DataFrame(X_test_poly_arr, columns=df_columns, index=X_test.index)

    return X_train_poly, X_test_poly

def df_target_encoding(
            df_train: pd.DataFrame,
            df_test: pd.DataFrame,
            y_train: pd.Series,
            encode_columns: List[str],
            target_type: Literal["continuous", "binary", "multiclass"] = "continuous",
            show_info: bool = True
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Целевое кодирование категориальных столбцов датафрейма

    Args:
        df_train (pd.DataFrame): Обучающий датафрейм
        df_test (pd.DataFrame): Тестовый датафрейм
        y_train (pd.Series): Обучающая челевая переменная
        encode_columns (List[str]): Имена стобцов для кодирования
        target_type (Literal[&quot;continuous&quot;, &quot;binary&quot;, &quot;multiclass&quot;], optional): Тип целевой переменной. Defaults to "continuous".
        show_info (bool, optional): Показывать статистики результата. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Обучающий и тестовый датафрейм с закодированными столбцами
    """
    train_inx = df_train.index
    test_inx = df_test.index

    df_train_cat = df_train[encode_columns]
    df_test_cat = df_test[encode_columns]

    encoder = TargetEncoder(target_type=target_type)

    train_encoded_arr = encoder.fit_transform(df_train_cat, y_train)
    test_encoded_arr = encoder.transform(df_test_cat)

    df_train_encoded = pd.DataFrame(
        data=train_encoded_arr,
        columns=encode_columns,
        index=train_inx)
    df_test_encoded = pd.DataFrame(
        data=test_encoded_arr,
        columns=encode_columns,
        index=test_inx)

    df_train[encode_columns] = df_train_encoded
    df_test[encode_columns] = df_test_encoded

    if show_info:
        print(df_train_encoded.shape, df_test_encoded.shape)
        display(df_train_encoded.describe().round(1))
        display(df_test_encoded.describe().round(1))

    return df_train, df_test
