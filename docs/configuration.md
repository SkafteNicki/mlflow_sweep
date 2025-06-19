The core object that needs to be created to use this package is a config file writing in YAML format that contains the
configuration for the hyperparameter optimization process. MLflow sweep utilizes this
[sweep](https://github.com/wandb/sweeps) library to sample hyperparameters. This package is developed by Weights and
Biases and Mlflow sweeps configuration format is therefore very similar to the one used by Weights and Biases. This
documentation part is therefore partly taken from [here](https://docs.wandb.ai/guides/sweeps/).

??? "Difference to Weights and Biases"

    The main difference to the Weights and Biases sweep configuration is the following:

    - The `command` field is a single string where parameters are specified using the `${parameter_name}` syntax. In
        Weights and Biases, the `command` field consist of a list of macros that determine how the command is run
        and parameters are passed to the command.

    - The `experiment_name` and `sweep_name` fields are used to create the experiment and sweep in MLflow. In Weights
        and Biases, this more or less corresponds to the `project` and `name` fields in the sweep.

    - Weights and Biases have a `entity` field for teams running sweeps, this is not present in MLflow sweeps.

    - Weights and Biases have a `early_terminate` field to stop runs that are not performing well, this is not present
        in MLflow sweeps (at the moment, will be added in the future).

    - Weights and Biases support three methods for hyperparameter optimization: `random`, `grid`, and `bayesian`.
        MLflow sweeps currently only support `random` and `grid`.

A minimal configuration file looks like this:

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
=== "poetry"

    ```yaml
    command: poetry run python example.py --learning-rate ${learning_rate} --batch-size ${batch_size}
    ```
=== "pipenv"

    ```yaml
    command: pipenv run python example.py --learning-rate ${learning_rate} --batch-size ${batch_size}
    ```

=== "uv"

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

=== "No hyphens"

    ```yaml
    command: uv run example.py learning_rate=${learning_rate} batch_size=${batch_size}
    ```

Currently, there are a couple of standard ways to pass parameters to your script, that we do not support yet:

* Environment variables: if you script loads in hyperparameters from environment variables, then this is not supported.

* JSON file: if you script loads in hyperparameters from a JSON file, then this is not supported.


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


| Value for distribution key | Description                                                                                                                                                       |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `constant`                 | Constant distribution. Must specify the constant value (`value`) to use.                                                                                          |
| `categorical`              | Categorical distribution. Must specify all valid values (`values`) for this hyperparameter.                                                                       |
| `int_uniform`              | Discrete uniform distribution on integers. Must specify `max` and `min` as integers.                                                                              |
| `uniform`                  | Continuous uniform distribution. Must specify `max` and `min` as floats.                                                                                          |
| `q_uniform`                | Quantized uniform distribution. Returns `round(X / q) * q` where `X` is uniform. `q` defaults to 1.                                                               |
| `log_uniform`              | Log-uniform distribution. Returns a value `X` between `exp(min)` and `exp(max)` such that the natural logarithm is uniformly distributed between `min` and `max`. |
| `log_uniform_values`       | Log-uniform distribution. Returns a value `X` between `min` and `max` such that `log(X)` is uniformly distributed between `log(min)` and `log(max)`.              |
| `q_log_uniform`            | Quantized log uniform. Returns `round(X / q) * q` where `X` is `log_uniform`. `q` defaults to 1.                                                                  |
| `q_log_uniform_values`     | Quantized log uniform. Returns `round(X / q) * q` where `X` is `log_uniform_values`. `q` defaults to 1.                                                           |
| `inv_log_uniform`          | Inverse log uniform distribution. Returns `X`, where `log(1/X)` is uniformly distributed between `min` and `max`.                                                 |
| `inv_log_uniform_values`   | Inverse log uniform distribution. Returns `X`, where `log(1/X)` is uniformly distributed between `log(1/max)` and `log(1/min)`.                                   |
| `normal`                   | Normal distribution. Return value is normally distributed with mean `mu` (default 0) and standard deviation `sigma` (default 1).                                  |
| `q_normal`                 | Quantized normal distribution. Returns `round(X / q) * q` where `X` is `normal`. `q` defaults to 1.                                                               |
| `log_normal`               | Log normal distribution. Returns a value `X` such that the natural logarithm `log(X)` is normally distributed with mean `mu` (default 0) and `sigma` (default 1). |
| `q_log_normal`             | Quantized log normal distribution. Returns `round(X / q) * q` where `X` is `log_normal`. `q` defaults to 1.                                                       |
