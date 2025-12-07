from typing import Any, Dict, List

from .base import TimeSeriesModelAgent


class ModelRegistry:
    """Registry for managing time series model agents."""

    def __init__(self) -> None:
        self._models: Dict[str, TimeSeriesModelAgent] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default models. Extended in Phase 3."""
        pass

    def register(self, model: TimeSeriesModelAgent) -> None:
        """Register a model instance."""
        self._models[model.name] = model

    def get_model(self, name: str) -> TimeSeriesModelAgent:
        """Get model by name. Raises KeyError if not found."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def get_compatible_models(self, data_characteristics: Dict[str, Any]) -> List[str]:
        """Filter models compatible with given data characteristics.

        Args:
            data_characteristics: Dict with keys:
                - is_multivariate: bool
                - has_exogenous: bool
                - n_samples: int
                - n_features: int

        Returns:
            List of compatible model names.
        """
        is_multivariate = data_characteristics.get("is_multivariate", False)
        has_exogenous = data_characteristics.get("has_exogenous", False)
        n_samples = data_characteristics.get("n_samples", 0)
        n_features = data_characteristics.get("n_features", 1)

        compatible = []
        for name, model in self._models.items():
            caps = model.capabilities

            # Multivariate check
            if is_multivariate and not caps.handles_multivariate:
                continue

            # Exogenous check
            if has_exogenous and not caps.handles_exogenous:
                continue

            # Min samples check
            if n_samples < caps.min_samples:
                continue

            # Max features check
            if n_features > caps.max_features:
                continue

            compatible.append(name)

        return compatible
