import os
import sys
import subprocess


def create_venv(venv_name=".venv"):
    """
    Создает виртуальное окружение.

    Args:
        venv_name (str): Имя папки для виртуального окружения. Defaults to '.venv'.
    """
    print("Создается виртуальное окружение...")
    subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
    print(f"Виртуальное окружение создано в папке {venv_name}.")

def activate_venv(venv_name=".venv"):
    """
    Возвращает путь к активатору виртуального окружения.

    Args:
        venv_name (str): Имя папки виртуального окружения. Defaults to 'venv'.

    Returns:
        str: Команда для активации виртуального окружения.
    """
    if os.name == "nt":  # Windows
        return os.path.join(venv_name, "Scripts", "activate")
    else:  # Linux/MacOS
        return f'source {os.path.join(venv_name, "bin", "activate")}'

def install_github_package(repo_url, branch="master", venv_name=".venv"):
    """
    Устанавливает пакет с GitHub в виртуальное окружение.

    Args:
        repo_url (str): URL репозитория GitHub.
        branch (str): Имя ветки. Defaults to 'main'.
        venv_name (str): Имя папки виртуального окружения. Defaults to 'venv'.
    """
    activate_command = activate_venv(venv_name)
    install_chardet_command = "pip install chardet"
    pip_install_command = f"pip install git+{repo_url}@{branch}"

    print("Активируется виртуальное окружение и устанавливается пакет...")
    try:
        subprocess.run(f"{activate_command} && {install_chardet_command} && {pip_install_command}", shell=True, check=True)
    except Exception:
        subprocess.run(f" /bin/bash -c '{activate_command} && {install_chardet_command} && {pip_install_command}'", shell=True, check=True)
    print("Пакет успешно установлен!")

if __name__ == "__main__":

    REPO_URL = "https://github.com/NazarovMichail/ds_nms.git"
    BRANCH = "master"

    # Создание виртуального окружения и установка пакета
    VENV_NAME = ".venv"
    create_venv(VENV_NAME)
    install_github_package(REPO_URL, BRANCH, VENV_NAME)
