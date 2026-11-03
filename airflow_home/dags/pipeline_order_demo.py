import pendulum
from airflow.sdk import dag, task

@dag(
    dag_id="pipeline_order_demo",
    schedule=None,
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    tags=["learning"],
)
def pipeline_order_demo():
    @task
    def pull_data():
        print("1. Veri indirme adımı")

    @task
    def transform_data():
        print("2. Veriyi dönüştürme adımı")

    @task
    def load_data():
        print("3. Veritabanına yükleme adımı")

    pull = pull_data()
    transform = transform_data()
    load = load_data()

    pull >> transform >> load

pipeline_order_demo()
