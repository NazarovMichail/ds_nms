from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest, kruskal
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_test_hists(sample_1: pd.Series,
                    sample_2: pd.Series) -> None:
    """Строит гистограммы для двух выборок.

    Args:
        sample_1 (pd.Series): Первая выборка.
        sample_2 (pd.Series):  Вторая выборка.
    """
    fig, ax = plt.subplots(1,2, figsize=[10, 5])
    sample_1.hist(ax=ax[0])
    sample_2.hist(ax=ax[1])
    ax[0].set_title('Выборка 1')
    ax[1].set_title('Выборка 2');

def kolmog_smirn_test(sample_1: pd.Series,
                    sample_2: pd.Series,
                    alpha: float=0.05) -> None:
    """Тест на то, принадлежат ли две выборки одному и тому же распределению

    Args:
        sample_1 (pd.Series): Выборка 1
        sample_2 (pd.Series): Выборка 2
        alpha (float, optional): Вероятность ошибочно отклонить нулевую гипотезу. Defaults to 0.05 (уровень значимости).
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
        alpha (float, optional): Вероятность ошибочно отклонить нулевую гипотезу. Defaults to 0.05 (уровень значимости).
    """

    statistic, p_value = kruskal(sample_1, sample_2)
    if p_value < alpha:
        print(f'p-value={p_value:.5f}')
        print("Гипотеза о равенстве распределения отвергается")
    else:
        print(f'p-value={p_value:.5f}')
        print("Данные могут быть взяты из одного распределения")


def adfuller_test(
    y: pd.Series,
    regression: Literal['n', 'c', 'ct', 'ctt'] = 'c',
    alpha: float = 0.05
    ) -> tuple:
    """Тест Дики-Фуллера (Augmented Dickey-Fuller Test, ADF), который проверяет временной ряд на стационарность.
    - H_0(нулевая гипотеза): ряд нестационарный, присутствует единичный корень.
    - H_1(альтернативная гипотеза): ряд стационарный,  единичного корня нет.

    Args:
        y (pd.Series): Временной ряд
        regression (Literal[&#39;n&#39;, &#39;c&#39;, &#39;ct&#39;, &#39;ctt&#39;], optional): Определяет, какие компоненты включать в тестовое уравнение. Defaults to 'c'.
        - 'c': Только константа
        - 'ct': Константа + линейный тренд
        - 'ctt': Константа + линейный + квадратичный тренд
        - 'n': Без константы и трендов

    Returns:
        tuple:
        - Тестовая статистика Дики-Фуллера,
        - p-value,
        - кол-во использованных лагов,
        - количество наблюдений,
        - Словарь критических значений для уровней значимости 1%, 5%, 10%
        - AIC, BIC или другие критерии, зависящие от параметра autolag
    """

    result  = adfuller(x=y, regression=regression)
    p_value = result[1]
    if p_value < alpha:
        print(f'ADF: p-value={p_value:.5f}')
        print("ADF: Гипотеза о нестационарности отвергается, ряд стационарный")
    else:
        print(f'ADF: p-value={p_value:.5f}')
        print("ADF: Гипотеза о нестационарности подтверждается, ряд нестационарный")

    return result

def kpss_test(
    y: pd.Series,
    regression: Literal[ 'c', 'ct'] = 'c',
    alpha: float = 0.05
    ) -> tuple:
    """KPSS-тест (Kwiatkowski-Phillips-Schmidt-Shin test) для проверки стационарности временного ряда.
    - H_0(нулевая гипотеза): ряд стационарный.
    - H_1(альтернативная гипотеза): ряд нестационарный.

    Args:
        y (pd.Series): Временной ряд
        regression (Literal[&#39;n&#39;, &#39;c&#39;, &#39;ct&#39;, &#39;ctt&#39;], optional): Определяет, какие компоненты включать в тестовое уравнение. Defaults to 'c'.
        - 'c': Только константа
        - 'ct': Константа + линейный тренд

    Returns:
        tuple:
        - Тестовая статистика KPSS,
        - p-value,
        - кол-во использованных лагов,
        - Словарь критических значений для уровней значимости 1%, 5%, 10%
    """

    result  = kpss(x=y, regression=regression)
    p_value = result[1]
    if p_value < alpha:
        print(f'KPSS: p-value={p_value:.5f}')
        print("KPSS: Гипотеза о стационарности отвергается, ряд нестационарный")
    else:
        print(f'KPSS: p-value={p_value:.5f}')
        print("KPSS: Гипотеза о стационарности подтверждается, ряд стационарный")

    return result


def stationarity_test(
    y: pd.Series,
    regression: Literal[ 'c', 'ct'] = 'c',
    alpha: float = 0.05
    ) -> dict:
    """Проведение теста на стационарность временного ряда:
    - Если ADF говорит, что ряд стационарен, а KPSS — нет, значит, ряд на грани стационарности.
    - Если оба теста говорят, что ряд стационарен → почти 100% уверенность.
    - Если оба теста говорят, что ряд нестационарен → нужно дифференцировать или удалять тренд.
    - Если ADF говорит, что ряд нестационарен, а KPSS — стационарен, значит, ряд тренд-стационарный стационарности.
        - 'c': Только константа
        - 'ct': Константа + линейный тренд
    Args:
        y (pd.Series): Временной ряд
        regression (Literal[ &#39;c&#39;, &#39;ct&#39;], optional): _description_. Defaults to 'c'.
        alpha (float, optional): Определяет, какие компоненты включать в тестовое уравнение. Defaults to 0.05.

    Returns:
        dict: Результаты теста ADF и KPSS
    """
    adf_result = adfuller_test(y=y,
                  regression=regression,
                  alpha=alpha)
    adf_p_value = adf_result[1]

    kpss_result = kpss_test(y=y,
                  regression=regression,
                  alpha=alpha)
    kpss_p_value = kpss_result[1]

    if adf_p_value < alpha and kpss_p_value < alpha:
        print("Результат ADF-теста и KPSS-теста: Ряд на грани стационарности")
    elif adf_p_value < alpha and kpss_p_value > alpha:
        print("Результат ADF-теста и KPSS-теста: Ряд стационарен")
    elif adf_p_value > alpha and kpss_p_value < alpha:
        print("Результат ADF-теста и KPSS-теста: Ряд нестационарен")
    else:
        print("Результат ADF-теста и KPSS-теста: Ряд тренд-стационарный. Для стационарности удалить тренд")

    result = {"ADF": adf_result,"KPSS": kpss_result}

    return result


def acf_pacf_plot(
    y: pd.Series,
    alpha=0.05,
    lags=36) -> None:
    """Построение графиков автокорреляции и частичной автокорреляции.
    - Автокорреляционная функция (ACF) помогает понять зависимость значений временного ряда от его прошлых значений.
    Выявляет сезонность, тренд и определения параметра q в модели ARIMA (MA-порядок).
    - Частичная автокорреляционная функция (PACF)помогает выявить,
    насколько текущие значения временного ряда зависят только от его предыдущих значений,
    исключая влияние промежуточных лагов
    Выявляет  параметр p в модели ARIMA (AR-порядок).
    Args:
        y (pd.Series): _description_
        regression (Literal[ &#39;c&#39;, &#39;ct&#39;], optional): _description_. Defaults to 'c'.
        alpha (float, optional): _description_. Defaults to 0.05.
    """
    f, ax = plt.subplots(1, 2, figsize=(10, 5))

    acf_fig = plot_acf(x=y, alpha=alpha, lags=lags, ax=ax[0])
    pacf_fig = plot_pacf(x=y, alpha=alpha, lags=lags, ax=ax[1])

    print("""Autocorrelation:
          - Быстро убывает после 1-2 лагов : Ряд стационарен
          - Периодические пики: присутствует сезонность
          - Резко обрывается после q-го лага: наличие скользящего среднего (MA) порядка q.""")
    print("""Partial Autocorrelation:
          - Резко обрывается после p-го лага : Ряд соответствует AR(p) (авторегрессия)
          - Периодические пики: присутствует сезонность
          - быстро обрывается, но ACF убывает постепенно: Указывает на ARIMA(p,d,0) модель.""")
