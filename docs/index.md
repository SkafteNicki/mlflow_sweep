# MLflow Sweep

<figure markdown="span">
  ![Image title](figures/logo.png){ width="300" }
  <figcaption>MLflow Sweep: A tool for hyperparameter optimization with MLflow</figcaption>
</figure>

MLflow sweep is a simple extension to the MLflow framework for running hyperparameter sweeps. In particular, it adds
the following functionality compared to running MLflow experiments directly:

- **Sweep configuration**: Define a sweep configuration file that specifies the command to run, the parameters to
    sweep over, and the sweep strategy. Support for multiple sweep strategies (grid search, random) and different
    parameter distributions (categorical, uniform, normal, log uniform).

- **No code change**: No changes to your code are required to run sweeps. You can use the same code you would use for
    running a single experiment.

- **Auto grouping**: Automatically groups runs by sweep name, making it easy to track and compare results.

- **Visualization**: Provides custom visualizations for sweep results, including parameter importance and performance
    metrics.

The package was heavily inspired by the sweep functionality of Weights & Biases and aims to provide a similar
experience for MLflow users. It is designed to be easy to use and integrate with existing MLflow workflows.

## 🔧 Installation

=== "pip"

    ```bash
    pip install mlflow-sweep
    ```

=== "uv"

    ```bash
    uv add mlflow-sweep
    ```

And that's it. You should now be able to see an additional `sweep` command in the MLflow CLI.

```txt
❯ mlflow --help
Usage: mlflow [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  artifacts    Upload, list, and download artifacts from an MLflow...
  db           Commands for managing an MLflow tracking database.
  deployments  Deploy MLflow models to custom targets.
  doctor       Prints out useful information for debugging issues with MLflow.
  experiments  Manage experiments.
  gc           Permanently delete runs in the `deleted` lifecycle stage.
  models       Deploy MLflow models locally.
  run          Run an MLflow project from the given URI.
  runs         Manage runs.
  sagemaker    Serve models on SageMaker.
  server       Run the MLflow tracking server.
  sweep        MLflow Sweep CLI commands.            <-- this is the new command
```

To learn more about the `sweep` command and how to configure and start a run, visit the following sections of the
documentation.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Quickstart guide on how to use MLflow sweep__

    ---

    A simple guide to get you started with MLflow sweep, including how to create a sweep configuration file and run
    sweeps

    [:octicons-arrow-right-24: Getting started](quickstart.md)

-   :material-file-document-plus:{ .lg .middle } __Configuration__

    ---

    A detailed description of the configuration file format, including how to specify commands, parameters, and sweep
    strategies. This is the core object that needs to be created to use this package.

    [:octicons-arrow-right-24: Configuration](configuration.md)

-   :material-api:{ .lg .middle } __API references__

    ---

    A detailed outline of modules, classes, and functions in the package. This is useful for developers who want to
    understand the internals of the package or extend it with custom functionality.

    [:octicons-arrow-right-24: API references](api_references.md)

</div>

## 🧑‍💻 Development setup

Mlflow sweep uses [uv](https://uv.dev) for development and packaging. To set up the development environment run the
following commands:

```bash
# Clone the repository
git clone https://github.com/yourusername/mlflow_sweep
cd mlflow_sweep

# Install using uv with development dependencies
uv sync --dev
```

py-[invoke](https://www.pyinvoke.org/) is used for running common tasks, such as running tests, building the package,
and generating documentation.

```txt
❯ uv run invoke --list
Available tasks:

    all        Run all tasks.
    check      Check code with pre-commit.
    clean      Clean up build artifacts.
    docs       Build the documentation.
    doctests   Run doctests.
    tests      Test and coverage.
```

## ❕ License

Package is licensed under Apache 2.0 license. See the LICENSE file for details. If you use this tool in your research,
please cite it as:

```bibtex
@misc{mlflow_sweep,
    author       = {Nicki Skafte Detlefsen},
    title        = {MLflow Sweep: A tool for hyperparameter optimization with MLflow},
    howpublished = {\url{https://github.com/SkafteNicki/mlflow_sweep}},
    year         = {2025}
}
```
