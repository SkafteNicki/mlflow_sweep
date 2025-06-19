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

A minimal configuration file looks like this:

```yaml
command:                      # Command to run the training script with parameters (required)
  uv run example.py
  --learning-rate ${learning_rate}
  --batch-size ${batch_size}
experiment_name: sweep-demo   # Name of the experiment
sweep_name: test-sweep        # Name of the sweep
method: random                # Method for hyperparameter optimization
metric:                       # Metric to optimize
  name: metric1
  goal: maximize
parameters:                   # Parameters to optimize (required)
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

Currently, MLflow sweep supports three methods for hyperparameter optimization: `bayes`, `random`, and `grid`. The
`bayes` method uses Bayesian optimization to sample hyperparameters, which is a more efficient way to explore the
hyperparameter space. The `random` method samples hyperparameters randomly from the specified distributions, while
the `grid` method samples hyperparameters from a grid of values.

## Metric configuration

The metric configuration is used to specify the metric that will be optimized during the sweep. This is only relevant
if you are using the `bayes` method for hyperparameter optimization. The metric is specified as a dictionary with the
following fields:

```
metric:                       # Metric to optimize
  name: metric1
  goal: maximize
```

where `goal` can be either `maximize` or `minimize`. The `name` field is the name of a metric which should be logged
during the run using `mlflow.log_metric`.

## Parameter Configuration

The most complex part of the configuration file is the `parameters` section, which defines the hyperparameters to be
optimized. It is a dictionary where each key is the name of a hyperparameter and then the options for that
hyperparameter

```yaml
parameters:
  parameter1:
    options for parameter1
  parameter2:
    options for parameter2
  ...
```

The following table lists the available options for each hyperparameter:

| Search Constraint | Description |
|-------------------|-------------|
| `values`          | Specifies all valid values for this hyperparameter. Compatible with grid. |
| `value`           | Specifies the single valid value for this hyperparameter. Compatible with grid. |
| `distribution`    | Specify a probability distribution. See the note following this table for information on default values. |
| `probabilities`   | Specify the probability of selecting each element of values when using random. |
| `min`, `max`      | `(int or float)` Maximum and minimum values. If int, for `int_uniform`-distributed hyperparameters. If float, for `uniform`-distributed hyperparameters. |
| `mu`              | `(float)` Mean parameter for `normal`- or `lognormal`-distributed hyperparameters. |
| `sigma`           | `(float)` Standard deviation parameter for `normal`- or `lognormal`-distributed hyperparameters. |
| `q`               | `(float)` Quantization step size for quantized hyperparameters. |

!!! info

    The following assumptions are made for the `distribution` key if not specified:

    * Set to `categorical` if `values` is specified.
    * Set to `constant` if `value` is specified.
    * Set to `int_uniform` if `min` and `max` are specified and `min` and `max` are integers.
    * Set to `uniform` if `min` and `max` are specified and `min` and `max` are floats.

??? "Values for distribution key"

    The following table lists the available values for the `distribution` key in the `parameters` section.

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

Here are examples of how to configure common hyperparameters:

=== "Learning rate"

    Learning rate is a common hyperparameter that is often optimized in machine learning models. Because values are on
    a logarithmic scale, we recommend using a log-uniform distribution to sample values.

    ```yaml
    parameters:
      learning_rate:
        distribution: log_uniform
        min: 1e-4
        max: 1e-2
    ```

=== "Batch size"

    Batch size is usually set based on the available memory, but if you want to optimize it you may consider using a
    categorical distribution to sample values from a list of possible batch sizes.

    ```yaml
    parameters:
      batch_size:
        distribution: categorical
        values: [16, 32, 64, 128]
    ```

=== "Number of layers"

    Number of layers is a common hyperparameter in deep learning models. It is usually an integer value, so you can use
    a discrete uniform distribution to sample values.

    ```yaml
    parameters:
      num_layers:
        distribution: int_uniform
        min: 1
        max: 10
    ```

=== "Dropout rate"

    Dropout rate is a common hyperparameter in deep learning models. It is usually a float value between 0 and 1, so
    you can use a uniform distribution to sample values.

    ```yaml
    parameters:
      dropout_rate:
        distribution: uniform
        min: 0.0
        max: 0.5
    ```

## Special cases

* If you have hyperparameters that are boolean values, most commonly the syntax for providing these as arguments would
  be `--hyperparameter` or `--no-hyperparameter`. In this case, you can use the `categorical` distribution with two
  strings:

  ```yaml
  command:
    uv run example.py ${hyperparameter}
  parameters:
    hyperparameter:
      distribution: categorical
      values: ["--hyperparameter", "--no-hyperparameter"]
  ```
