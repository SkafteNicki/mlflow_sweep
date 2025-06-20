import joblib
import mlflow
import typer
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
    output: str = typer.Option("rf_model.pkl", help="Path to save trained model"),
):
    """Example command to train a RandomForestClassifier on the breast cancer dataset."""
    # Load dataset
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    typer.echo(f"Train Accuracy: {train_acc:.4f}")
    typer.echo(f"Test Accuracy:  {test_acc:.4f}")

    joblib.dump(model, output)
    typer.echo(f"Model saved to {output}")

    with mlflow.start_run(run_name="random_forest"):
        mlflow.set_tag("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("bootstrap", bootstrap)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_artifact(output)


if __name__ == "__main__":
    app()
