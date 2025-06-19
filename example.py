import random
import time

import mlflow
import typer

app = typer.Typer()


@app.command()
def main(learning_rate: float = 0.001, batch_size: int = 32, dropout: bool = False):
    """
    Example command that takes learning rate and batch size as arguments.
    """
    typer.echo(f"Learning Rate: {learning_rate}")
    typer.echo(f"Batch Size: {batch_size}")

    with mlflow.start_run():
        # mlflow.log_param("learning_rate", learning_rate)
        # mlflow.log_param("batch_size", batch_size)

        for i in range(random.randint(1, 5)):
            mlflow.log_metric("metric1", random.uniform(0, 1), step=i)
            mlflow.log_metric("metric2", random.uniform(0, 1), step=i)
            time.sleep(random.uniform(0.1, 1))


if __name__ == "__main__":
    app()
