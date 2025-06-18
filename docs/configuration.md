The core object that needs to be created to use this package is a config file writing in YAML
format that contains the configuration for the hyperparameter optimization process.

```yaml
command:                      # Command to run the training script with parameters
  uv run example.py
  --learning-rate ${learning_rate}
  --batch-size ${batch_size}
experiment_name: sweep-demo   # Name of the experiment
sweep_name: test-sweep        # Name of the sweep
method: random
metric:                       # Metric to optimize
  name: metric1
  goal: maximize
parameters:                   # Parameters to optimize
  learning_rate:
    distribution: uniform
    min: 0.001
    max: 0.1
  batch_size:
    values: [16, 32, 64, 128]
run_cap: 10                   # Maximum number of runs to execute
```

The `experiment_name` corresponds to the name of the experiment where the sweep will be created.
It corresponds directly to `mlflow.create_experiment` in the MLflow API. By default, this is set to the `Default`
namespace. `sweep_name` is the name of the sweep that will be created in the experiment. It is used to group the runs of
the sweep together and corresponds to the `mlflow.start_run` in the MLflow API. The `run_cap` is the maximum number of
runs that will be executed in the sweep. If not specified, it will be set to 10.

The remaining fields are explained in more details below.

## Command configuration

The `command` field is a string that contains the command to run the training script with the parameters that will be
optimized. The parameters are specified using the `${parameter_name}` syntax, where `parameter_name` is the name of the
parameter defined in the `parameters` section of the configuration file. Example of how to configure based on which
package manager you are using

=== "Standard Python"

    ```yaml
    command: python example.py --learning-rate ${learning_rate} --batch-size ${batch_size}
    ```
=== "Poetry"

    ```yaml
    command: poetry run python example.py --learning-rate ${learning_rate} --batch-size ${batch_size}
    ```
=== "Pipenv"

    ```yaml
    command: pipenv run python example.py --learning-rate ${learning_rate} --batch-size ${batch_size}
    ```

=== "UV"

    ```yaml
    command: uv run example.py --learning-rate ${learning_rate} --batch-size ${batch_size}
    ```

In the same way, depending on how you pass parameters to your script you should adjust the command accordingly

=== "Positional arguments"

    ```yaml
    command: uv run example.py ${learning_rate} ${batch_size}
    ```

=== "Named arguments"

    ```yaml
    command: uv run example.py --learning-rate ${learning_rate} --batch-size ${batch_size}
    ```

=== "Hydra configuration"

    ```yaml
    command: uv run example.py --config-name config.yaml learning_rate=${learning_rate} batch_size=${batch_size}
    ```

===

## Method configuration

Currently, MLflow sweep supports two methods for hyperparameter optimization: `random` and `grid`. The `random` method
samples hyperparameters randomly from the specified distributions, while the `grid` method samples hyperparameters from
a grid of values.

## Metric configuration

## Parameter Configuration

=== "Categorical variables"

    something something

=== "Uniform"

    something something

=== "Normal"

    something something

=== "Log Uniform"

    something something
