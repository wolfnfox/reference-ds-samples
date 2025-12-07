from typing import Any, Dict

import numpy as np
import pandas as pd


class DataIngestionAgent:
    """Agent for loading and validating time series data."""

    IQR_MULTIPLIER = 1.5

    def load_and_validate(self, excel_path: str) -> pd.DataFrame:
        """Load Excel file and validate basic structure.

        Args:
            excel_path: Path to Excel file.

        Returns:
            Cleaned DataFrame.

        Raises:
            FileNotFoundError: File not found.
            ValueError: Empty file or no columns.
        """
        try:
            df = pd.read_excel(excel_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {excel_path}")

        if df.empty:
            raise ValueError(f"Empty file: {excel_path}")

        if len(df.columns) == 0:
            raise ValueError(f"No columns in file: {excel_path}")

        return df

    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report.

        Args:
            df: Input DataFrame.

        Returns:
            Dict with quality metrics:
                - n_rows, n_cols
                - missing_percentage: per column
                - column_types: dtype mapping
                - outliers: columns with outlier counts
        """
        n_rows, n_cols = df.shape

        # Missing percentages
        missing_pct = (df.isnull().sum() / n_rows * 100).to_dict()

        # Column types
        col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Outliers via IQR for numeric columns
        outliers = self._detect_outliers(df)

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "missing_percentage": missing_pct,
            "column_types": col_types,
            "outliers": outliers,
        }

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers using IQR method.

        Returns:
            Dict mapping column name to outlier count.
        """
        outliers: Dict[str, int] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = df[col].dropna()
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.IQR_MULTIPLIER * iqr
            upper = q3 + self.IQR_MULTIPLIER * iqr
            count = ((col_data < lower) | (col_data > upper)).sum()
            if count > 0:
                outliers[col] = int(count)

        return outliers
