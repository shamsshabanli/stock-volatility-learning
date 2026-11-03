import pendulum
from airflow.sdk import dag, task

@dag(
    dag_id="first_learning_dag",
    schedule=None,
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    tags=["learning"],
)
def first_learning_dag():
    @task
    def say_hello():
        print("Merhaba Airflow!")

    say_hello()

first_learning_dag()
