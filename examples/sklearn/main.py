import mlflow
import typer
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

app = typer.Typer(help="Train a RandomForestClassifier on the breast cancer dataset")


def none_str_to_none(value: str) -> None | str:
    """Convert 'None' string to None."""
    return None if value.lower() == "none" else value


@app.command()
def train(
    n_estimators: int = typer.Option(100, help="Number of trees"),
    criterion: str = typer.Option("gini", help="Criterion for splitting"),
    max_depth: int | None = typer.Option(None, help="Max depth of the trees"),
    min_samples_split: int = typer.Option(2, help="Min samples required to split a node"),
    min_samples_leaf: int = typer.Option(1, help="Min samples required at a leaf node"),
    max_features: str | None = typer.Option(
        None, help="Number of features to consider for best split", parser=none_str_to_none
    ),
    bootstrap: bool = typer.Option(True, help="Use bootstrap samples"),
):
    """Example command to train a RandomForestClassifier on the breast cancer dataset."""
    # Load dataset
    data = datasets.load_breast_cancer()
    x, y = data.data, data.target

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42,
    )

    scores = cross_val_score(model, x, y, cv=5, scoring="accuracy")
    typer.echo(f"Cross-validation scores: {scores}")

    with mlflow.start_run(run_name="random_forest"):
        mlflow.set_tag("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("bootstrap", bootstrap)
        mlflow.log_metric("mean_accuracy", scores.mean())
        mlflow.log_metric("std_accuracy", scores.std())


if __name__ == "__main__":
    app()
