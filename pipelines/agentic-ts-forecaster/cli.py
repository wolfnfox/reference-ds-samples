from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from config import OrchestratorConfig, PipelineConfig
from pipeline import ForecastPipeline

app = typer.Typer()
console = Console()


@app.command()
def forecast(
    excel_path: Path = typer.Argument(..., help="Path to Excel file"),
    orchestrator: str = typer.Option("rule-based", "--orchestrator", "-o"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    context: str = typer.Option("", "--context", "-x", help="Business context"),
    target: str = typer.Option("target", "--target", "-t", help="Target column"),
    date_col: str = typer.Option("date", "--date", "-d", help="Date column"),
):
    """Run time series forecasting pipeline"""
    console.print(f"[bold]Forecast: {excel_path}[/bold]")

    if config_path:
        config = PipelineConfig.from_yaml(str(config_path))
    else:
        config = PipelineConfig(
            orchestrator=OrchestratorConfig(type=orchestrator),
            target_column=target,
            date_column=date_col,
        )

    pipeline = ForecastPipeline(config)
    result = pipeline.run(str(excel_path), context)

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    table = Table(title="Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Best Model", result.get("best_model", "N/A"))
    for k, v in result.get("metrics", {}).items():
        table.add_row(k.upper(), f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)

    if result.get("report_path"):
        console.print(f"Report: {result['report_path']}")

@app.command()
def init():
    """Initialize project structure"""
    console.print("Creating project directories...")
    
    directories = [
        "orchestrator",
        "models", 
        "agents",
        "config",
        "tests",
        "outputs"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        (Path(dir_name) / "__init__.py").touch()
    
    console.print("[green]✓[/green] Project structure created")

if __name__ == "__main__":
    app()
