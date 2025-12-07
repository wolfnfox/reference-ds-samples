from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd

@dataclass
class ModelCapabilities:
    handles_multivariate: bool
    handles_exogenous: bool
    handles_seasonality: bool
    requires_stationary: bool
    min_samples: int
    max_features: int
    best_for: List[str]

@dataclass
class ModelResult:
    model_name: str
    predictions: pd.DataFrame
    metrics: Dict[str, float]
    training_time: float
    interpretability: Dict[str, Any]
    config: Dict[str, Any]
    metadata: Dict[str, Any]

class TimeSeriesModelAgent(ABC):
    """Base class for all time series models"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        pass
    
    @abstractmethod
    def train(
        self, 
        train_data: pd.DataFrame,
        target_column: str,
        exogenous_columns: List[str] = None,
        config: Dict[str, Any] = None
    ) -> None:
        pass
    
    @abstractmethod
    def predict(
        self, 
        horizon: int,
        exogenous_future: pd.DataFrame = None
    ) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def evaluate(
        self,
        test_data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def interpret(self) -> Dict[str, Any]:
        pass
    
    def run_full_pipeline(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        exogenous_columns: List[str] = None,
        config: Dict[str, Any] = None
    ) -> ModelResult:
        """Standardized full pipeline - implements common logic"""
        import time
        
        start_time = time.time()
        
        self.train(train_data, target_column, exogenous_columns, config)
        predictions = self.predict(
            horizon=len(test_data),
            exogenous_future=test_data[exogenous_columns] if exogenous_columns else None
        )
        metrics = self.evaluate(test_data, target_column)
        interpretability = self.interpret()
        
        training_time = time.time() - start_time
        
        return ModelResult(
            model_name=self.name,
            predictions=predictions,
            metrics=metrics,
            training_time=training_time,
            interpretability=interpretability,
            config=config or {},
            metadata={"capabilities": self.capabilities}
        )
