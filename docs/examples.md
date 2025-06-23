Below are some examples of how to the main script for running a MLflow experiment could look like and what the
corresponding sweep configuration file could look like.

## Dummy example

??? example "main.py"

    ```python title="main.py"
    --8<-- "examples/dummy/main.py"
    ```

??? example "sweep.yaml"

    ```yaml title="sweep.yaml"
    --8<-- "examples/dummy/sweep.yaml"
    ```

## Scikit-learn example

??? example "main.py"

    ```python title="main.py"
    --8<-- "examples/sklearn/main.py"
    ```

??? example "sweep.yaml"

    ```yaml title="sweep.yaml"
    --8<-- "examples/sklearn/sweep.yaml"
    ```

## MLflow project example

??? example "main.py"

    ```python title="main.py"
    --8<-- "examples/mlflow_project/main.py"
    ```

??? example "sweep.yaml"

    ```yaml title="sweep.yaml"
    --8<-- "examples/mlflow_project/sweep.yaml"
    ```

??? example "MLproject"

    ```yaml title="MLproject"
    --8<-- "examples/mlflow_project/MLproject"
    ```
