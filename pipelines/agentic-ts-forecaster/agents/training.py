from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from models import ModelRegistry, ModelResult

console = Console()


class TrainingAgent:
    """Agent for parallel model training."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def train_parallel(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        model_names: List[str],
        target_column: str,
        date_column: str,
        exogenous_columns: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[ModelResult]:
        """Train multiple models in parallel.

        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame for evaluation
            model_names: List of model names to train
            target_column: Target variable column name
            date_column: Date column name
            exogenous_columns: Optional exogenous feature columns
            config: Optional model configuration dict

        Returns:
            List of ModelResult for successful models
        """
        results: List[ModelResult] = []

        def train_model(name: str) -> Optional[ModelResult]:
            try:
                model = self._registry.get_model(name)
                return model.run_full_pipeline(
                    train_data=train_data,
                    test_data=test_data,
                    target_column=target_column,
                    date_column=date_column,
                    exogenous_columns=exogenous_columns,
                    config=config,
                )
            except Exception as e:
                console.print(f"[red]Model {name} failed: {e}[/red]")
                return None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training models...", total=len(model_names))

            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(train_model, name): name for name in model_names
                }

                for future in as_completed(futures):
                    name = futures[future]
                    result = future.result()
                    if result:
                        results.append(result)
                        console.print(f"[green]{name}[/green] trained ({result.training_time:.1f}s)")
                    progress.advance(task)

        return results
