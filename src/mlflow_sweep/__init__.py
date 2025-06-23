__version__ = "0.1.0"


def cli():
    """Wrapper CLI around the standard MLflow CLI to add sweep commands."""
    import click
    from mlflow.cli import cli as mlflow_cli

    from mlflow_sweep.commands import finalize_command, init_command, run_command

    @mlflow_cli.group()
    def sweep():
        """MLflow Sweep CLI commands."""
        pass

    @sweep.command("init")
    @click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
    def init(config_path):
        """Initialize a new sweep configuration."""
        init_command(config_path)

    @sweep.command("run")
    @click.option(
        "--sweep-id",
        default="",
        type=str,
        help="ID of the sweep to run (optional if not specified will use the most recent initialized sweep)",
    )
    def run(sweep_id):
        """Start a sweep agent."""
        run_command(sweep_id)

    @sweep.command("finalize")
    @click.option(
        "--sweep-id",
        default="",
        type=str,
        help="ID of the sweep to finalize (optional if not specified will use the most recent initialized sweep)",
    )
    def finalize(sweep_id):
        """Finalize a sweep."""
        finalize_command(sweep_id)

    return mlflow_cli()
