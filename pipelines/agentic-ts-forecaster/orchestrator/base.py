from abc import ABC, abstractmethod
from typing import  Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ModelSelectionDecision:
    selected_models: List[str]
    rationale: str
    expected_winner: str
    expected_winner_reason: str
    confidence: float
    metadata: Dict[str, Any]

class OrchestratorBase(ABC):
    """Base class for all orchestrators"""
    
    @abstractmethod
    def select_models(
        self,
        data_profile: Dict[str, Any],
        business_context: str,
        available_models: List[str]
    ) -> ModelSelectionDecision:
        """Select which models to evaluate"""
        pass
    
    @abstractmethod
    def analyze_results(
        self,
        model_results: List[Any],  # List[ModelResult]
        data_profile: Dict[str, Any],
        business_context: str
    ) -> Dict[str, Any]:
        """Analyze model outputs and generate insights"""
        pass
    
    @abstractmethod
    def generate_report(
        self,
        analysis: Dict[str, Any],
        model_results: List[Any],
        market_research: Optional[str] = None
    ) -> str:
        """Generate final report"""
        pass
