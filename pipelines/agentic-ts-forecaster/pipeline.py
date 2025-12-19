import json
from pathlib import Path
from typing import Any, Dict

from rich.console import Console
from rich.status import Status
from sklearn.model_selection import train_test_split

from agents import DataIngestionAgent, EDAAgent, TrainingAgent
from config import PipelineConfig
from models import ModelRegistry
from orchestrator import OrchestratorFactory

console = Console()


class ForecastPipeline:
    """Main forecasting pipeline orchestrating all agents."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._orchestrator = OrchestratorFactory.create(
            config.orchestrator.type,
            api_key=config.orchestrator.claude_api_key,
            model=config.orchestrator.claude_model,
            model_name=config.orchestrator.local_model_name,
        )
        self._registry = ModelRegistry()
        self._data_agent = DataIngestionAgent()
        self._eda_agent = EDAAgent()
        self._training_agent = TrainingAgent(self._registry)

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self, excel_path: str, business_context: str = "") -> Dict[str, Any]:
        """Execute full forecasting pipeline.

        Args:
            excel_path: Path to input Excel file
            business_context: Optional business context for analysis

        Returns:
            Dict with pipeline results including best_model, metrics, report_path
        """
        output_dir = Path(self._config.output_dir)

        # Step 1: Data ingestion
        with Status("[bold blue]Loading data...", console=console):
            df = self._data_agent.load_and_validate(excel_path)
        console.print(f"[green]Loaded {len(df)} rows[/green]")

        # Step 2: Quality report
        with Status("[bold blue]Generating quality report...", console=console):
            quality_report = self._data_agent.generate_quality_report(df)
            self._save_json(output_dir / "quality_report.json", quality_report)
        console.print("[green]Quality report saved[/green]")

        # Step 3: EDA profiling
        with Status("[bold blue]Profiling data...", console=console):
            data_profile = self._eda_agent.profile_data(
                df, self._config.target_column, self._config.date_column
            )
            self._save_json(output_dir / "eda_profile.json", data_profile)
        console.print("[green]EDA profile saved[/green]")

        # Step 4: Train/test split
        train_data, test_data = train_test_split(
            df, test_size=self._config.test_size, shuffle=False
        )
        console.print(f"Split: {len(train_data)} train, {len(test_data)} test")

        # Step 5: Model selection
        available_models = self._registry.get_compatible_models({
            "has_exogenous": bool(self._config.exogenous_columns),
            "n_samples": len(train_data),
            "n_features": len(self._config.exogenous_columns) + 1,
        })

        if self._config.force_models:
            model_names = self._config.force_models
        else:
            with Status("[bold blue]Selecting models...", console=console):
                decision = self._orchestrator.select_models(
                    data_profile, business_context, available_models
                )
            model_names = decision.selected_models
            console.print(f"[green]Selected: {model_names}[/green]")

        # Step 6: Parallel training
        console.print("[bold]Training models...[/bold]")
        results = self._training_agent.train_parallel(
            train_data=train_data,
            test_data=test_data,
            model_names=model_names,
            target_column=self._config.target_column,
            date_column=self._config.date_column,
            exogenous_columns=self._config.exogenous_columns or None,
        )

        if not results:
            console.print("[red]All models failed[/red]")
            return {"error": "All models failed"}

        # Step 7: Result analysis
        with Status("[bold blue]Analyzing results...", console=console):
            analysis = self._orchestrator.analyze_results(
                results, data_profile, business_context
            )
        console.print(f"[green]Best model: {analysis.get('best_model')}[/green]")

        # Step 8: Report generation
        if self._config.generate_report:
            with Status("[bold blue]Generating report...", console=console):
                report = self._orchestrator.generate_report(analysis, results, None)
                report_path = output_dir / "forecast_report.md"
                report_path.write_text(report)
            console.print(f"[green]Report: {report_path}[/green]")
        else:
            report_path = None

        return {
            "best_model": analysis.get("best_model"),
            "metrics": analysis.get("metrics", {}),
            "insights": analysis.get("insights", []),
            "report_path": str(report_path) if report_path else None,
            "output_dir": str(output_dir),
        }

    def _save_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Save dict as JSON, handling non-serializable types."""
        def default(o):
            if hasattr(o, "tolist"):
                return o.tolist()
            if hasattr(o, "__dict__"):
                return o.__dict__
            return str(o)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=default)
