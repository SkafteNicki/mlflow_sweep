from invoke import Context, task


@task
def docs(ctx: Context) -> None:
    """Build the documentation."""
    ctx.run("uv run mkdocs serve")


@task
def clean(ctx: Context) -> None:
    """Clean up build artifacts."""
    ctx.run("rm -rf build dist *.egg-info", echo=True, pty=True)
    ctx.run("find . -name '*.pyc' -delete", echo=True, pty=True)
    ctx.run("find . -name '__pycache__' -delete", echo=True, pty=True)
    ctx.run("rm -rf .pytest_cache", echo=True, pty=True)
    ctx.run("rm -rf .ruff_cache", echo=True, pty=True)
    ctx.run("rm -f .coverage", echo=True, pty=True)
    ctx.run("find . -type d -name 'mlruns' -exec rm -rf {} +", echo=True, pty=True)


@task
def doctests(ctx: Context) -> None:
    """Run doctests."""
    ctx.run("uv run pytest src/mlflow_sweep -v", echo=True, pty=True)


@task
def tests(ctx: Context) -> None:
    """Test and coverage."""
    ctx.run("uv run coverage run -m pytest tests/ -v", echo=True, pty=True)
    ctx.run("uv run coverage report -i -m", echo=True, pty=True)


@task
def check(ctx: Context) -> None:
    """Check code with pre-commit."""
    ctx.run("uv run pre-commit run --all-files", echo=True, pty=True)


@task(pre=[check, doctests, tests, clean])
def all(ctx: Context) -> None:
    """Run all tasks."""
    pass
