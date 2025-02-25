<img src='img/logo.gif'>

___
# <center>Описание проекта
`ds_nms` библиотека для аналитики данных и машинного обучения, включающая модули для предобработки данных, отбора признаков, обучения моделей, тюнинга гиперпараметров и проведения статистического анализа. Она позволяет автоматизировать ключевые этапы работы с данными и моделями машинного обучения, обеспечивая эффективные инструменты для анализа и оптимизации.

___
# <center>Содержание
- [Структура репозитория](#1)
- [Установка библиотеки](#2)
- [Описание модулей](#3)
- [Примеры использования](#4)

___
# <center><a id=1>Структура репозитория</a>

```
📦 Репозиторий (ds_nms)
├── 📂 ds_nms/                   [Основной пакет библиотеки]
│   ├── 📄 data_proc.py          [Обработка и предобработка данных]
│   ├── 📄 feature_select.py     [Отбор значимых признаков]
│   ├── 📄 model_train.py        [Обучение моделей]
│   ├── 📄 model_tune.py         [Оптимизация гиперпараметров]
│   ├── 📄 model_tune_params.py  [Параметры моделей]
│   ├── 📄 stat_tests.py         [Статистические тесты]
│   ├── 📄 utils_io.py           [Вспомогательные утилиты]
│   └── 📄 __init__.py           [Файл инициализации пакета]
│
├── 📂 install/                  [Скрипт установки пакета]
│   └── 📄 install_ds_nms.py
│
├── 📂 examples/                 [Примеры использования]
│   ├── 📄 example1.ipynb        [Jupyter Notebook с примерами работы]
│   ├── 📄 example2.ipynb
│   └── ...
│
├── 📄 requirements.txt          [Файл с зависимостями (Ubuntu)]
├── 📄 requirements_win.txt      [Файл с зависимостями (Windows)]
├── 📄 .env                      [Файл с переменными среды (MLFlow)]
└── 📄 README.md
```
___
# <center><a id=2>Установка библиотеки</a>
Основные требования:
- Операционная система: `Ubuntu 22.04` / Windows
- `Python: 3.10.12` / `Python: 3.12.3`

## Вариант 1. Установка при помощи скрипта
Установка осуществляется через скрипт install_ds_nms.py, который автоматически создает виртуальное окружение и устанавливает пакет из GitHub.

Шаги установки:
1. Из директории репозитория [install/](https://github.com/NazarovMichail/ds_nms/tree/master/install) скачать в рабочую директорию файл [install_ds_nms.py](https://github.com/NazarovMichail/ds_nms/blob/master/install/install_ds_nms.py)
2. Выполните команду:

 ```bash
python3 install_ds_nms.py
```
- Создается виртуальное окружение .venv
- Установится пакет ds_nms из GitHub-репозитория

## Вариант 2. Ручная установка
1. Создать виртуальное окружение
```bash
python3 -m venv .venv
```
2. Активировать окружение
```bash
source .venv/bin/activate
```
3. Установить пакет
```bash
pip install git+https://github.com/NazarovMichail/ds_nms.git@master
```
## Файл .env
Файл .env - файл с переменными среды для подключения к базе данных MLFlow.

Содержание файла:
- USER_ML = <Имя пользователя Posgresql>
- PASS_ML = <Пароль Posgresql>
- HOST_ML = <Хост Posgresql>
- PORT_ML = <Порт Posgresql>
- BD_ML = <Название БД Posgresql>
___
# <center><a id=3>Описание модулей</a>

## Модуль data_proc

Предназначен для обработки и предобработки данных перед анализом и обучением моделей машинного обучения.

- get_duplicated_df: Поиск и обработка дубликатов
- features_separate: Разделяет признаки на категориальные и числовые на основе порога уникальных значений
- get_stratified_df: Добавляет в датафрейм стратификационные группы по квартилям указанного признака
- df_scaling: Масштабирует числовые данные с использованием StandardScaler, MinMaxScaler, Normalizer, RobustScaler
- drop_outliers_iso: Удаляет выбросы из данных с использованием метода IsolationForest
- drop_outliers_tuk: Удаляет выбросы по методу Тьюки (IQR)
- get_pca: Преобразует группы признаков с использованием PCA и добавляет результат в датафрейм
- get_VIF: Вычисляет коэффициент инфляции дисперсии (VIF) для анализа мультиколлинеарности
- plot_corrmatrix: Строит корреляционную матрицу и вычисляет ее ранг и детерминант
- get_polyfeatures: Генерирует полиномиальные признаки
- df_target_encoding: Выполняет целевое кодирование категориальных признаков с использованием TargetEncoder

## Модуль feature_select

Предназначен для отбора значимых признаков в наборах данных

- get_selected_features: Выполняет отбор признаков с использованием RFE или SFS, возвращая новый набор данных с выбранными переменными.
- get_best_n_features: Оптимизирует количество признаков с помощью Optuna, проводя кросс-валидацию и выбирая наилучший набор переменных.
- save_selected_features: Выполняет отбор признаков для нескольких датасетов и моделей, сохраняя результаты

## Модуль model_train

 Предназначен для обучения моделей машинного обучения, оценки их качества и анализа важности признаков.

- get_feature_importance_df: Возвращает таблицу важности признаков для обученной модели
- get_importances_barplot: Строит график важности признаков, используя коэффициенты модели и SHAP-значения.
- get_feat_importance_arima: Анализирует важность признаков в модели SARIMAX
- get_feature_contrib: Анализирует вклад признаков в предсказания линейных моделей
- plot_ts_pred: Строит график предсказанных и фактических значений для временных рядов.
- get_prediction: Возвращает датафрейм с метриками качества модели и массив предсказаний.
- train_cv: Обучает модель с использованием различных схем кросс-валидации (KFold, LeaveOneOut, StratifiedKFold, TimeSeriesSplit)
- arima_train: Автоматический подбор параметров и обучение SARIMAX модели для временных рядов.
- arima_predict: Выполняет предсказание временного ряда с использованием обученной модели SARIMAX.

## Модуль model_tune

Предназначен для оптимизации гиперпараметров моделей машинного обучения с использованием Optuna.

- get_optimize_params: Выполняет подбор оптимальных гиперпараметров модели с использованием Optuna.
- get_optimize_results: Запускает полный процесс оптимизации и последующего обучения модели с лучшими параметрами. Возвращает обученную модель, метрики и предсказания.
- get_optimize_several_results: Оптимизирует несколько моделей на различных датасетах, сохраняя результаты.
- get_best_study_params: Определяет лучшие гиперпараметры, отфильтровывая результаты по заданным критериям
- optuna_plot: Визуализирует Pareto-графики, важность параметров и зависимость метрик от гиперпараметров.
- mlflow_save_results: Логирует метрики и обученные модели в MLflow
- get_prediction_df: Создает итоговый датафрейм с предсказаниями и ошибками модели

## Модуль model_tune_params

Предназначен для управления гиперпараметрами моделей, используемых в процессе обучения и оптимизации. Служит справочником параметров, которые можно изменять и передавать в model_tune для автоматической оптимизации.

- Класс ModelsParams: Хранит параметры моделей для оптимизации, включая диапазоны значений и базовые параметры. Поддерживаемые модели:
    - K-Nearest Neighbors (KNN)
    - Lasso, Ridge, ElasticNet
    - Huber Regressor
    - Stochastic Gradient Descent (SGD)
    - Support Vector Regressor (SVR)
    - Decision Trees (DT)
    - Random Forest (RF)
    - Extra Trees (EXTR)
    - XGBoost (XGB)
    - LightGBM (LGBM)
    - Stacking Regressor
    - Bayesian Ridge
    - Tweedie Regressor
    - Theil-Sen Regressor
    - ARD Regression
    - Passive Aggressive Regressor (PAR)

- set_param: Позволяет изменять параметры модели перед их передачей в model_tune
## Модуль stat_tests

Предназначен для проведения статистических тестов и анализа данных.

- plot_test_hists: Строит гистограммы для двух выборок, позволяя визуально сравнить их распределения.
- kolmog_smirn_test: Тест Колмогорова-Смирнова – проверяет, принадлежат ли две выборки одному распределению.
- kraskel_wallis_test: Тест Краскела-Уоллиса – анализирует статистически значимые различия между независимыми выборками.
- adfuller_test: Тест Дики-Фуллера (ADF) – проверяет временной ряд на стационарность.
- kpss_test: KPSS-тест – альтернатива ADF для проверки стационарности временного ряда.
- stationarity_test: Объединяет результаты ADF и KPSS, предоставляя комплексную оценку стационарности.
- acf_pacf_plot: Строит автокорреляционную функцию (ACF) и частичную автокорреляционную функцию (PACF).

## Модуль utils_io

Предназначен для работы с файлами и данными, включая сохранение и загрузку файлов в различных форматах (pickle, json), управление сервером MLflow, а также очистку логов в PostgreSQL.

- save_data: Сохраняет данные (словарь с объектами) в указанную директорию в форматах pkl или json.
- load_data: Загружает данные из указанной директории в форматах pkl или json.
- mlflow_server_start: Запускает сервер MLflow для логирования экспериментов
- mlflow_run_delete: Удаляет логи MLflow из базы данных PostgreSQL

___
# <center><a id=4>Примеры использования</a>

- [data_proc_test.ipynb](https://github.com/NazarovMichail/ds_nms/blob/master/examples/data_proc_test.ipynb): Тестирование функций предобработки данных, включая обнаружение выбросов, масштабирование, кодирование категориальных признаков и анализ корреляции.
    - data_proc
    - stat_tests
    - utils_io

- [data_ts_test.ipynb](https://github.com/NazarovMichail/ds_nms/blob/master/examples/data_ts_test.ipynb): Анализ и обработка временных рядов, включая разложение сезонности, обучение моделей и оценку качества прогнозирования
    - model_train

- [feature_contribution.ipynb](https://github.com/NazarovMichail/ds_nms/blob/master/examples/feature_contribution.ipynb): Анализ вклада признаков в предсказания моделей с использованием методов интерпретируемости, таких как коэффициенты линейных моделей и SHAP-значения.
    - model_train

- [feature_select_test.ipynb](https://github.com/NazarovMichail/ds_nms/blob/master/examples/feature_select_test.ipynb): Проверка методов отбора признаков
    - feature_select
    - utils_io

- [model_tune_test.ipynb](https://github.com/NazarovMichail/ds_nms/blob/master/examples/model_tune_test.ipynb): Оптимизация гиперпараметров моделей с использованием Optuna, оценка качества моделей и сохранение лучших конфигураций.
    - model_tune
    - model_tune_params
    - model_train
    - utils_io

- [training_test.ipynb](https://github.com/NazarovMichail/ds_nms/blob/master/examples/training_test.ipynb): Проверка функций обучения моделей, расчета метрик и визуализации предсказаний.
    - model_train
    - utils_io
