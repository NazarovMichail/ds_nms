import os
import subprocess
from dotenv import load_dotenv


load_dotenv()
USER_ML = os.getenv("USER_ML")
PASS_ML = os.getenv("PASS_ML")
BD_ML = os.getenv("BD_ML")
S3 = os.getenv("S3")

def run_mlflow_server():
    cmd = [
        "mlflow",
        "server",
        "--backend-store-uri",
        f"postgresql://{USER_ML}:{PASS_ML}@localhost:5432/{BD_ML}",
        "--default-artifact-root",
        f"s3://{S3}",
        "--serve-artifacts"
    ]

    subprocess.run(cmd)

if __name__ == "__main__":
    run_mlflow_server()
