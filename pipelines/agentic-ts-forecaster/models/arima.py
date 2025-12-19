from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base import ModelCapabilities, TimeSeriesModelAgent
from ..helpers.dictionary import get_or_default


class ARIMAAgent(TimeSeriesModelAgent):
    """ARIMA model agent using pmdarima for automatic order selection."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._target_column: Optional[str] = None

    @property
    def name(self) -> str:
        return "ARIMA"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            handles_multivariate=False,
            handles_exogenous=True,
            handles_seasonality=True,
            requires_stationary=False,
            min_samples=30,
            max_features=10,
            best_for=["univariate", "trend", "seasonality"],
        )

    def train(
        self,
        train_data: pd.DataFrame,
        target_column: str,
        date_column: str,
        exogenous_columns: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train ARIMA model using auto_arima for order selection."""
        config = config or {}
        self._target_column = target_column

        y = train_data[target_column].values
        exog = train_data[exogenous_columns].values if exogenous_columns else None

        # Auto ARIMA parameters
        seasonal = get_or_default(config, "seasonal", bool, True)
        m = get_or_default(config, "m", int, 12)  # Seasonal period
        max_p = get_or_default(config, "max_p", int, 3)
        max_q = get_or_default(config, "max_q", int, 3)
        max_d = get_or_default(config, "max_d", int, 2)
        max_P = get_or_default(config, "max_P", int, 2)
        max_Q = get_or_default(config, "max_Q", int, 2)
        max_D = get_or_default(config, "max_D", int, 1)

        self._model = auto_arima(
            y, exogenous=exog, scale_exog=True,
            start_p=0, d=None, start_q=0, max_p=max_p, max_d=max_d, max_q=max_q,
            start_P=0, D=None, start_Q=0, max_P=max_P, max_D=max_D, max_Q=max_Q,
            max_order=5, m=m, seasonal=seasonal, stationary=False,
            information_criterion='aic', alpha=0.05,
            test='kpss', seasonal_test='ocsb',
            stepwise=True, n_jobs=1, method='lbfgs', maxiter=50,
            trace=True, error_action='ignore', suppress_warnings=True
        )

    def predict(
        self, horizon: int, exogenous_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Generate forecast for given horizon."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        exog = exogenous_future.values if exogenous_future is not None else None
        forecast = self._model.predict(n_periods=horizon, exogenous=exog)

        return pd.DataFrame({"prediction": forecast})

    def evaluate(self, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_true = test_data[target_column].values
        y_pred = self._model.predict(n_periods=len(test_data))

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # MAPE (avoid division by zero)
        mask = y_true != 0
        if mask.any():
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = float("inf")

        return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

    def interpret(self) -> Dict[str, Any]:
        """Return model interpretation details."""
        if self._model is None:
            return {"error": "Model not trained"}

        return {
            "order": self._model.order,
            "seasonal_order": self._model.seasonal_order,
            "aic": float(self._model.aic()),
            "bic": float(self._model.bic()),
            "params": {k: float(v) for k, v in self._model.params().items()},
        }
