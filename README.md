# MLflow Sweep

MLflow Sweep is an extension of MLflow that adds functionality for hyperparameter optimization.

## 🌟 Features

- 🚀 Seamless integration with MLflow for experiment tracking
- 📊 Comprehensive hyperparameter sweeping capabilities
- 📈 Visualization of sweep results
- 🔄 Support for multiple sweep strategies (grid search, random, Bayesian)
- 🧩 Extensible architecture for custom sweep configurations

## 🔧 Installation

```bash
pip install mlflow-sweep
```

### Development setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mlflow_sweep
cd mlflow_sweep

# Install using uv with development dependencies
uv sync --dev
```

## 📋 Requirements

- Python 3.11+
- MLflow

In particular, because Python uses a *last install wins* strategy, you should ensure that the `mlflow-sweep` package is
installed after MLflow. The reason for this is that `mlflow-sweep` extends the MLflow CLI with a new command
`mlflow sweep` and this command will not be available if `mlflow` was the last package installed/updated.

## 📖 Usage

Start by creating a sweep configuration file (e.g., `sweep_config.yaml`) that defines what command to run, the
parameters to sweep over and the sweep strategy. Here is an example configuration:

```yaml
command:
  uv run example.py
  --learning-rate ${learning_rate}
  --batch-size ${batch_size}
experiment_name: sweep-demo
sweep_name: test-sweep
method: random
parameters:
  learning_rate:
    distribution: uniform
    min: 0.001
    max: 0.1
  batch_size:
    values: [16, 32, 64, 128]
run_cap: 10
```

The configuration file uses close to the same syntax as Weights & Biases (W&B)
[sweeps](https://docs.wandb.ai/guides/sweeps/sweep-config-keys/#parameters), in particular the `parameters` section
is exactly the same.

Then use the `mlflow sweep init` command to initialize the sweep:

```bash
mlflow sweep init sweep_config.yaml
```

This will create a sweep (a
[parent run](https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/part1-child-runs)
in MLflow) which is just a MLflow run with the sweep configuration saved as an artifact. It will return the ID of the
sweep run.

Then you can use the `mlflow sweep run` command to start the sweep:

```bash
mlflow sweep run --sweep-id=<sweep_id>
```

The `--sweep-id` argument is the ID of the sweep run created in the previous step. It is an optional argument and if
not provided we will look for the most recent initialized sweep run in the current directory. The `mlflow sweep run`
command can be executed in parallel to parallelize the search process. The process will either stop when the `run_cap`
is reached or when all combinations of the parameters have been tried (only applicable for grid search).

Finally, you can use the `mlflow sweep finalize` command to finalize the sweep:

```bash
mlflow sweep finalize --sweep-id=<sweep_id>
```

This will mark the sweep as completed and do a final analysis of the results. The final analysis will be saved as
artifacts to the parent sweep run.

## ❕ License

Package is licensed under Apache 2.0 license. See the LICENSE file for details.
If you use this tool in your research, please cite it as:

```bibtex
@misc{mlflow_sweep,
    author       = {Nicki Skafte Detlefsen},
    title        = {MLflow Sweep: A tool for hyperparameter optimization with MLflow},
    howpublished = {\url{https://github.com/SkafteNicki/mlflow_sweep}},
    year         = {2025}
}
```
