from typing import List, Any, Dict
import pickle
import json
import os
import subprocess
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.sql import text


def save_data(
        file_dict: Dict[str, Any],
        sub_dir: str = '',
        directory: str = 'data/',
        format: str = 'pkl') -> None:
    """
    Сохраняет данные из словаря в указанную директорию.

    Args:
        file_dict (Dict[str, Any]): Словарь с именами файлов и данными для сохранения.
        sub_dir (str): Поддиректория для сохранения файлов.
        directory (str, optional): Основная директория для сохранения. Defaults to 'data/'.
        format (str, optional): Формат сохранения ('pkl' или 'json'). Defaults to 'pkl'.

    Returns:
        None
    """
    final_dir = os.path.join(directory, sub_dir)
    os.makedirs(final_dir, exist_ok=True)

    for file_name, data in file_dict.items():
        file_path = os.path.join(final_dir, f"{file_name}.{format}")
        try:
            if format == 'pkl':
                with open(file_path, 'wb') as file:
                    pickle.dump(data, file)

            elif format == 'json':
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
            else:
                raise ValueError(f"Неподдерживаемый формат: {format}")
            print(f"Файл {file_path} успешно сохранён.")
        except Exception as error:
            print(f"Ошибка при сохранении файла {file_name}: {error}")

def load_data(
        # file_lst: List[str],
        directory: str = 'data/',
        sub_dir: str = '',
        load_format: str = 'pkl') -> List[Any]:
    """
    Загружает данные из указанной директории.

    Args:
        directory (str, optional): Основная директория. Defaults to 'data/'.
        sub_dir (str): Поддиректория, где находятся файлы.
        format (str, optional): Формат файлов ('pkl' или 'json'). Defaults to 'pkl'.

    Returns:
        List[Any]: Список загруженных данных.
    """
    loaded_dict = {}
    final_dir = os.path.join(directory, sub_dir)
    file_lst = os.listdir(final_dir)

    for file_name in file_lst:
        file_path = os.path.join(directory, f'{sub_dir}/{file_name}')
        try:
            if load_format == 'pkl':
                with open(file_path, 'rb') as file:
                    loaded_file = pickle.load(file)
            elif load_format == 'json':
                with open(file_path, 'r', encoding='utf-8') as file:
                    loaded_file = json.load(file)
            else:
                raise ValueError(f"Неподдерживаемый формат: {load_format}")

            file_name_cut = file_name.replace(f'.{load_format}', '')
            loaded_dict[file_name_cut] = loaded_file
            print(f"Файл {file_path} успешно загружен.")

        except FileNotFoundError as e:
            print(f"Файл {file_path} не найден.| {e}")
        except Exception as error:
            print(f"Ошибка при загрузке файла {file_path}: {error}")

    return loaded_dict

def save_split_description(df_initial: pd.DataFrame,
                            initial_columns: List[str],
                            target: pd.Series,
                            df_name: str,
                            directory: str='data',
                            ) -> None:
    """Сохраняет датафрейм с выбранными индексами целевой переменной и выбранными столбцами

    Args:
        df_initial (pd.DataFrame): Исходный датафрейм
        initial_columns (List[str]): Столбцы для нового датафрейма
        target (pd.Series): Целевая переменная
        df_name (str): Имя нового датафрейма
        directory (str, optional): Директория для сохранения нового датафрейма. Defaults to 'data'.

    Raises:
        ValueError: Ошибка, если колонки отсутсвуют в исходном датафрейме
    """
    missing_columns = set(initial_columns) - set(df_initial.columns)
    if missing_columns:
        raise ValueError(f"Следующие колонки отсутствуют в датафрейме: {missing_columns}")

    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"{df_name}_descr.pkl")

    df_description = df_initial.loc[target.index, initial_columns]
    df_description.to_pickle(file_path)

    print(f"Файл {file_path} сохранен!")


def mlflow_server_start():
    """Запуск сервера MLFlow
    """
    load_dotenv()
    USER_ML = os.getenv("USER_ML")
    PASS_ML = os.getenv("PASS_ML")
    BD_ML = os.getenv("BD_ML")

    cmd = [
        "mlflow",
        "server",
        "--backend-store-uri",
        f"postgresql://{USER_ML}:{PASS_ML}@localhost:5432/{BD_ML}",
        "--default-artifact-root",
        "mlruns/",
        "--serve-artifacts"
    ]

    try:
        subprocess.Popen(cmd)
        print("Сервер MLFlow запущен...")
    except Exception as e:
        print(e)

def mlflow_run_delete() -> None:

    """ Удаляет  логи MLFlow из БД postgres"""
    load_dotenv()
    USER_ML = os.getenv("USER_ML")
    PASS_ML = os.getenv("PASS_ML")
    BD_ML = os.getenv("BD_ML")
    engine = create_engine(f"postgresql://{USER_ML}:{PASS_ML}@localhost:5432/{BD_ML}")

    queries = """
        DELETE FROM experiment_tags WHERE experiment_id=ANY(SELECT experiment_id FROM experiments where lifecycle_stage='deleted');
        DELETE FROM latest_metrics WHERE run_uuid=ANY(SELECT run_uuid FROM runs WHERE experiment_id=ANY(SELECT experiment_id FROM experiments where lifecycle_stage='deleted'));
        DELETE FROM metrics WHERE run_uuid=ANY(SELECT run_uuid FROM runs WHERE experiment_id=ANY(SELECT experiment_id FROM experiments where lifecycle_stage='deleted'));
        DELETE FROM tags WHERE run_uuid=ANY(SELECT run_uuid FROM runs WHERE experiment_id=ANY(SELECT experiment_id FROM experiments where lifecycle_stage='deleted'));
        DELETE FROM params WHERE run_uuid=ANY(SELECT run_uuid FROM runs WHERE experiment_id=ANY(SELECT experiment_id FROM experiments where lifecycle_stage='deleted'));
        DELETE FROM runs WHERE experiment_id=ANY(SELECT experiment_id FROM experiments where lifecycle_stage='deleted');
        DELETE FROM experiments where lifecycle_stage='deleted';
    """

    connection = engine.connect()

    transaction = connection.begin()
    result = connection.execute(text(queries))
    transaction.commit()

    connection.close()
    print("Логи MLFlow из БД postgres удалены!")
