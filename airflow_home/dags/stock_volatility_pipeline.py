import logging
import subprocess
import sys
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

# DAG dosyası airflow_home/dags içinde;
# parents[2] proje ana klasörünü bulur.
PROJECT_DIR = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def run_script(task_name):
    """Run one pipeline script and surface a clear error in the Airflow log."""
    script_path = PROJECT_DIR / f"{task_name}.py"
    logger.info("%s görevi başladı: %s", task_name, script_path.name)
    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_DIR,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.exception("%s görevi başarısız oldu", task_name)
        raise
    logger.info("%s görevi başarıyla tamamlandı", task_name)

@dag(
    dag_id="stock_volatility_pipeline",
    # Hafta içi her gün 00:00 UTC (Bakü saatiyle 04:00).
    schedule="0 0 * * 1-5",
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    tags=["stock-volatility"],
)
def stock_volatility_pipeline():
    @task
    def pull_data():
        run_script("pull_data")

    @task
    def transform_data():
        run_script("transform_data")

    @task
    def load_data():
        run_script("load_data")

    pull = pull_data()
    transform = transform_data()
    load = load_data()

    pull >> transform >> load

stock_volatility_pipeline()
