from typing import Any, Dict, List, Optional

from .base import ModelSelectionDecision, OrchestratorBase


class RuleBasedOrchestrator(OrchestratorBase):
    """Deterministic rule-based orchestrator for model selection and analysis."""

    MAX_MODELS = 4
    HIGH_SEASONALITY_THRESHOLD = 0.5
    LARGE_DATASET_THRESHOLD = 200

    def select_models(
        self,
        data_profile: Dict[str, Any],
        business_context: str,
        available_models: List[str],
    ) -> ModelSelectionDecision:
        """Select models based on data characteristics using predefined rules."""
        selected = []
        rationale_parts = []

        n_samples = data_profile.get("n_samples", 0)
        is_multivariate = data_profile.get("is_multivariate", False)
        has_seasonality = data_profile.get("has_seasonality", False)
        seasonality_strength = data_profile.get("seasonality_strength", 0.0)

        # Rule 1: Always include ARIMA as baseline
        if "ARIMA" in available_models:
            selected.append("ARIMA")
            rationale_parts.append("ARIMA included as baseline")

        # Rule 2: Large dataset - add deep learning models
        if n_samples > self.LARGE_DATASET_THRESHOLD:
            if is_multivariate and "PatchTSMixer" in available_models:
                selected.append("PatchTSMixer")
                rationale_parts.append(
                    f"PatchTSMixer added for multivariate data with {n_samples} samples"
                )
            elif "LSTM" in available_models:
                selected.append("LSTM")
                rationale_parts.append(f"LSTM added for {n_samples} samples")

        # Rule 3: High seasonality - add Prophet
        if has_seasonality or seasonality_strength > self.HIGH_SEASONALITY_THRESHOLD:
            if "Prophet" in available_models and "Prophet" not in selected:
                selected.append("Prophet")
                rationale_parts.append("Prophet added for seasonal patterns")

        # Limit to max models
        selected = selected[: self.MAX_MODELS]

        # Fallback if no models selected
        if not selected:
            selected = available_models[: self.MAX_MODELS]
            rationale_parts.append("Fallback: using first available models")

        # Determine expected winner
        if is_multivariate and "PatchTSMixer" in selected:
            expected_winner = "PatchTSMixer"
            expected_reason = "Best suited for multivariate time series"
        elif has_seasonality and "Prophet" in selected:
            expected_winner = "Prophet"
            expected_reason = "Optimized for seasonal data"
        elif "ARIMA" in selected:
            expected_winner = "ARIMA"
            expected_reason = "Strong baseline for univariate forecasting"
        else:
            expected_winner = selected[0] if selected else ""
            expected_reason = "Default selection"

        return ModelSelectionDecision(
            selected_models=selected,
            rationale="; ".join(rationale_parts),
            expected_winner=expected_winner,
            expected_winner_reason=expected_reason,
            confidence=0.7,  # Rule-based has moderate confidence
            metadata={"selection_rules": rationale_parts},
        )

    def analyze_results(
        self,
        model_results: List[Any],
        data_profile: Dict[str, Any],
        business_context: str,
    ) -> Dict[str, Any]:
        """Analyze model results and generate insights."""
        if not model_results:
            return {
                "best_model": None,
                "insights": ["No models were evaluated"],
                "confidence_level": "low",
            }

        # Sort by RMSE (lower is better)
        sorted_results = sorted(
            model_results, key=lambda r: r.metrics.get("rmse", float("inf"))
        )

        best = sorted_results[0]
        best_rmse = best.metrics.get("rmse", None)

        if best_rmse is None:
            raise ValueError("Best model does not have RMSE metric, cannot compute insights.")
        if best_rmse == 0:
            raise ValueError("Best model has RMSE of zero, cannot compute insights.")

        insights = [f"{best.model_name} achieved lowest RMSE ({best_rmse:.4f})"]

        # Compare with other models
        if len(sorted_results) > 1:
            second = sorted_results[1]
            second_rmse = second.metrics.get("rmse", None)
            improvement = ((second_rmse - best_rmse) / second_rmse * 100) if second_rmse else 0
            insights.append(
                f"{best.model_name} outperformed {second.model_name} by {improvement:.1f}%"
            )

        # Training time insight
        fastest = min(model_results, key=lambda r: r.training_time)
        if fastest.model_name != best.model_name:
            insights.append(
                f"{fastest.model_name} was fastest to train ({fastest.training_time:.1f}s)"
            )

        # Confidence based on margin
        if len(sorted_results) > 1:
            margin = (sorted_results[1].metrics.get("rmse", 0) - best_rmse) / best_rmse
            confidence_level = "high" if margin > 0.1 else "medium" if margin > 0.05 else "low"
        else:
            confidence_level = "medium"

        return {
            "best_model": best.model_name,
            "best_metrics": best.metrics,
            "model_ranking": [r.model_name for r in sorted_results],
            "insights": insights,
            "confidence_level": confidence_level,
        }

    def generate_report(
        self,
        analysis: Dict[str, Any],
        model_results: List[Any],
        market_research: Optional[str] = None,
    ) -> str:
        """Generate markdown report summarizing results."""
        lines = ["# Forecast Analysis Report", ""]

        # Best Model Section
        lines.append("## Best Model")
        best = analysis.get("best_model", "N/A")
        lines.append(f"**Recommended**: {best}")
        lines.append(f"**Confidence**: {analysis.get('confidence_level', 'N/A')}")
        lines.append("")

        # Metrics
        if "best_metrics" in analysis:
            metrics = analysis["best_metrics"]
            lines.append("### Performance Metrics")
            for metric, value in metrics.items():
                lines.append(f"- **{metric.upper()}**: {value:.4f}")
            lines.append("")

        # Model Comparison Section
        lines.append("## Model Comparison")
        if model_results:
            lines.append("| Model | RMSE | MAE | R² | Training Time |")
            lines.append("|-------|------|-----|-----|---------------|")
            for r in sorted(model_results, key=lambda x: x.metrics.get("rmse", float("inf"))):
                rmse = r.metrics.get("rmse", 0.0)
                rmse = rmse if isinstance(rmse, (float, int)) else 0
                mae = r.metrics.get("mae", 0.0)
                mae = mae if isinstance(mae, (float, int)) else 0
                r2 = r.metrics.get("r2", 0.0)
                r2 = r2 if isinstance(r2, (float, int)) else 0
                lines.append(f"| {r.model_name} | {rmse:.4f} | {mae:.4f} | {r2:.4f} | {r.training_time:.1f}s |")
            lines.append("")

        # Insights Section
        lines.append("## Key Insights")
        for insight in analysis.get("insights", []):
            lines.append(f"- {insight}")
        lines.append("")

        # Business Implications
        lines.append("## Business Implications")
        lines.append(f"- The {best} model provides the most accurate forecasts for this dataset")
        lines.append("- Consider retraining periodically as new data becomes available")
        lines.append("")

        # Considerations
        lines.append("## Considerations")
        lines.append("- Results based on historical data; actual performance may vary")
        lines.append("- Model selection was rule-based; consider domain expertise for final decision")
        if analysis.get("confidence_level") == "low":
            lines.append("- Low confidence suggests models performed similarly; ensemble may help")
        lines.append("")

        return "\n".join(lines)
