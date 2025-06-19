


Start by creating a sweep configuration file (e.g., `sweep_config.yaml`) that defines what command to run, the
parameters to sweep over and the sweep strategy. Here is an example configuration:

```yaml title="sweep_config.yaml"
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

You can read more on [this page](configuration.md) about the configuration file and the available options.

## Sweep init

After having created the sweep configuration file, you can initialize a sweep using the `mlflow sweep init` command.
The command takes the path to the sweep configuration file as an argument:

```bash
mlflow sweep init sweep_config.yaml
```

You should see an output similar to this:

```
❯ mlflow sweep init sweep_config.yaml
Initializing sweep with configuration:
SweepConfig(
    command='uv run example.py --learning-rate ${learning_rate} --batch-size ${batch_size}',
    experiment_name='sweep-demo',
    sweep_name='test-sweep',
    method=<SweepMethodEnum.random: 'random'>,
    metric=MetricConfig(name='metric1', goal=<GoalEnum.maximize: 'maximize'>),
    parameters={'learning_rate': {'distribution': 'uniform', 'min': 0.001, 'max': 0.1}, 'batch_size': {'values': [16, 32, 64, 128]}},
    run_cap=10
)
2025/06/19 11:45:04 INFO mlflow.tracking.fluent: Experiment with name 'sweep-demo' does not exist. Creating a new experiment.
Sweep initialized with ID: 7556efd6d1fd46f0b3d893000e6f287a
```

This will create a sweep (a
[parent run](https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/part1-child-runs)
in MLflow) which is just a MLflow run with the sweep configuration saved as an artifact. The last line of the output
is the ID of the sweep run, which you will need for the next steps. If you try to spin op the MLflow UI, you will
see the sweep run listed there under the experiment you specified in the configuration file

<figure markdown="span">
  ![Image title](figures/init.png){ width="700" }
  <figcaption>After running the init command a single mlflow is created.</figcaption>
</figure>

## Sweep run

Then you can use the `mlflow sweep run` command to start the sweep:

```bash
mlflow sweep run --sweep-id=<sweep_id>
```

The `--sweep-id` argument is the ID of the sweep run created in the previous step. It is an optional argument and if
not provided we will look for the most recent initialized sweep run in the current directory. The `mlflow sweep run`
command can be executed in parallel to parallelize the search process. The process will either stop when the `run_cap`
is reached or when all combinations of the parameters have been tried (only applicable for grid search).

<figure markdown="span">
  ![Image title](figures/parallel.png){ width="700" }
  <figcaption>Example parallel execution. Each terminal is executing the run command and will report back to the main
  sweep run to synchronize sampled hyperparameters and metrics.</figcaption>
</figure>

If you try to spin op the MLflow UI, you should now see multiple child runs under the sweep run, each representing a
single run of the command with a different set of hyperparameters. Each child run will contain all parameters, metrics,
artifacts etc. that you are normally logging in your MLflow runs.

<figure markdown="span">
  ![Image title](figures/nested.png){ width="700" }
  <figcaption>Runs are automatically nested under the sweep run for easy overview.</figcaption>
</figure>

## Sweep finalize

Finally, you can use the `mlflow sweep finalize` command to finalize the sweep:

```bash
mlflow sweep finalize --sweep-id=<sweep_id>
```

You should see an output similar to this:

```txt
❯ uvr mlflow sweep finalize
              Feature Importance and Correlation for metric1
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Parameter     ┃ Importance ┃ Permutation Importance ┃ Pearson ┃ Spearman ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ learning_rate │     0.7913 │                 0.9954 │  0.1822 │   0.1636 │
├───────────────┼────────────┼────────────────────────┼─────────┼──────────┤
│ batch_size    │     0.2087 │                 0.1476 │  0.1912 │   0.1729 │
└───────────────┴────────────┴────────────────────────┴─────────┴──────────┘
```

This is an analysis of the parameter importance, permutation importance and correlation of the parameters with the
metric you specified in the sweep configuration file. These results are visualized and these visualizations are saved as
artifacts to the parent sweep run for future reference.
