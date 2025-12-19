from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats


class EDAAgent:
    """Agent for exploratory data analysis and profiling."""

    DEFAULT_SEASONALITY_THRESHOLD = 0.3
    DEFAULT_TREND_PVALUE = 0.05
    DEFAULT_SEASONAL_LAGS = [12, 24]

    def __init__(
        self,
        seasonality_threshold: float = DEFAULT_SEASONALITY_THRESHOLD,
        trend_pvalue: float = DEFAULT_TREND_PVALUE,
        seasonal_lags: List[int] = DEFAULT_SEASONAL_LAGS,
    ) -> None:
        self.seasonality_threshold = seasonality_threshold
        self.trend_pvalue = trend_pvalue
        self.seasonal_lags = seasonal_lags if seasonal_lags else self.DEFAULT_SEASONAL_LAGS

    def profile_data(
        self, df: pd.DataFrame, target_column: str, date_column: str
    ) -> Dict[str, Any]:
        """Profile time series data for orchestrator model selection.

        Args:
            df: Input DataFrame.
            target_column: Name of target variable.
            date_column: Name of date/time column.

        Returns:
            Dict with profile matching orchestrator expectations:
                - n_samples, n_features
                - has_exogenous
                - has_seasonality, seasonality_strength
                - trend_detected
                - missing_percentage
        """
        n_samples = len(df)

        # Feature columns (exclude date and target)
        feature_cols = [c for c in df.columns if c not in [date_column, target_column]]
        n_features = len(feature_cols)

        has_exogenous = n_features > 0

        # Get target series
        target_series = df[target_column].dropna()

        # Seasonality detection
        has_seasonality, seasonality_strength = self._analyze_seasonality(target_series)

        # Trend detection
        trend_detected = self.detect_trend(target_series)

        # Missing data
        missing_pct = df[target_column].isnull().sum() / n_samples * 100

        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "has_exogenous": has_exogenous,
            "has_seasonality": has_seasonality,
            "seasonality_strength": seasonality_strength,
            "trend_detected": trend_detected,
            "missing_percentage": missing_pct,
        }

    def _analyze_seasonality(self, series: pd.Series) -> tuple[bool, float]:
        """Analyze seasonality and return detection flag + strength.

        Returns:
            Tuple of (has_seasonality, seasonality_strength).
            Strength is max |ACF| across seasonal lags.
        """
        if len(series) < max(self.seasonal_lags) + 1:
            return False, 0.0

        acf_values = []
        for lag in self.seasonal_lags:
            if lag < len(series):
                acf = series.autocorr(lag=lag)
                if not np.isnan(acf):
                    acf_values.append(abs(acf))

        if not acf_values:
            return False, 0.0

        max_acf = max(acf_values)
        has_seasonality = max_acf > self.seasonality_threshold

        return has_seasonality, max_acf

    def detect_seasonality(self, series: pd.Series) -> bool:
        """Check for seasonality using ACF at seasonal lags.

        Args:
            series: Target time series.

        Returns:
            True if significant autocorrelation at seasonal lags.
        """
        has_seasonality, _ = self._analyze_seasonality(series)
        return has_seasonality

    def detect_trend(self, series: pd.Series) -> bool:
        """Check for trend using linear regression.

        Args:
            series: Target time series.

        Returns:
            True if slope is statistically significant.
        """
        if len(series) < 3:
            return False

        x = np.arange(len(series))
        y = series.values

        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            return False

        x_clean = x[mask]
        y_clean = y[mask]

        result = stats.linregress(x_clean, y_clean)
        return result.pvalue < self.trend_pvalue
