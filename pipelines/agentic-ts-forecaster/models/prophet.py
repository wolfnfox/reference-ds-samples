import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base import ModelCapabilities, TimeSeriesModelAgent


class ProphetAgent(TimeSeriesModelAgent):
    """Prophet model agent for time series forecasting."""

    def __init__(self) -> None:
        self._model: Optional[Prophet] = None
        self._date_column: Optional[str] = None
        self._target_column: Optional[str] = None
        self._exogenous_columns: Optional[List[str]] = None
        self._train_df: Optional[pd.DataFrame] = None

    @property
    def name(self) -> str:
        return "Prophet"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            handles_multivariate=False,
            handles_exogenous=True,
            handles_seasonality=True,
            requires_stationary=False,
            min_samples=30,
            max_features=10,
            best_for=["seasonality", "holidays", "missing data"],
        )

    def train(
        self,
        train_data: pd.DataFrame,
        target_column: str,
        date_column: str,
        exogenous_columns: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train Prophet model."""
        config = config or {}
        self._date_column = date_column
        self._target_column = target_column
        self._exogenous_columns = exogenous_columns

        # Prepare Prophet format: ds, y
        df = pd.DataFrame({
            "ds": pd.to_datetime(train_data[date_column]),
            "y": train_data[target_column].values,
        })

        # Add exogenous columns
        if exogenous_columns:
            for col in exogenous_columns:
                df[col] = train_data[col].values

        self._train_df = df

        # Suppress Prophet logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        # Initialize Prophet with config
        self._model = Prophet(
            yearly_seasonality=config.get("yearly_seasonality", "auto"),
            weekly_seasonality=config.get("weekly_seasonality", "auto"),
            daily_seasonality=config.get("daily_seasonality", "auto"),
            seasonality_mode=config.get("seasonality_mode", "additive"),
        )

        # Add regressors
        if exogenous_columns:
            for col in exogenous_columns:
                self._model.add_regressor(col)

        self._model.fit(df)

    def predict(
        self, horizon: int, exogenous_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Generate forecast for given horizon."""
        if self._model is None or self._train_df is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Create future dataframe
        future = self._model.make_future_dataframe(periods=horizon, freq="D")

        # Add exogenous columns to future
        if self._exogenous_columns and exogenous_future is not None:
            # Combine historical and future exogenous values
            for col in self._exogenous_columns:
                historical = self._train_df[col].values
                future_vals = exogenous_future[col].values
                future[col] = np.concatenate([historical, future_vals])

        forecast = self._model.predict(future)

        # Return only the forecast portion
        predictions = forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        return predictions.rename(columns={"yhat": "prediction"})

    def evaluate(self, test_data: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        if self._model is None or self._train_df is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_true = test_data[target_column].values

        # Generate predictions for test period
        future = self._model.make_future_dataframe(periods=len(test_data), freq="D")

        if self._exogenous_columns:
            for col in self._exogenous_columns:
                historical = self._train_df[col].values
                if col in test_data.columns:
                    future_vals = test_data[col].values
                    future[col] = np.concatenate([historical, future_vals])

        forecast = self._model.predict(future)
        y_pred = forecast.tail(len(test_data))["yhat"].values

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # MAPE
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
            "trend": "additive",
            "seasonality_mode": self._model.seasonality_mode,
            "seasonalities": list(self._model.seasonalities.keys()),
            "regressors": list(self._model.extra_regressors.keys()) if self._model.extra_regressors else [],
        }
