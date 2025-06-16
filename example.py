import typer
import mlflow
import random
import time

app = typer.Typer()

@app.command()
def main(learning_rate: float = 0.001, batch_size: int = 32, parent_run_id: str | None = None):
    """
    Example command that takes learning rate and batch size as arguments.
    """
    typer.echo(f"Learning Rate: {learning_rate}")
    typer.echo(f"Batch Size: {batch_size}")

    with mlflow.start_run(parent_run_id=parent_run_id):
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        # Here you would typically run your training code
        # For demonstration, we just log a dummy metric
        mlflow.log_metric("metric1", random.uniform(0, 1))
        mlflow.log_metric("metric2", random.uniform(0, 1))

        time.sleep(100 * random.uniform(0.1, 1.0))  # Simulate a long-running process

if __name__ == "__main__":
    app()