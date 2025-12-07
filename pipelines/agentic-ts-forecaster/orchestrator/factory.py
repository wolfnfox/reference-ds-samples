from typing import Any

from .base import OrchestratorBase
from .rule_based_orchestrator import RuleBasedOrchestrator


class OrchestratorFactory:
    """Factory for creating orchestrator instances."""

    @staticmethod
    def create(orchestrator_type: str, **kwargs: Any) -> OrchestratorBase:
        """Create orchestrator instance based on type.

        Args:
            orchestrator_type: One of 'claude', 'local', 'rule-based'
            **kwargs: Type-specific configuration (api_key, model_name, etc.)

        Returns:
            OrchestratorBase instance

        Raises:
            ValueError: Unknown orchestrator type
        """
        if orchestrator_type == "rule-based":
            return RuleBasedOrchestrator()

        elif orchestrator_type == "local":
            # Lazy import to avoid dependency if not using local
            try:
                from .local_orchestrator import LocalSLMOrchestrator
            except ImportError as e:
                raise ImportError("Ollama not installed. Run: pip install ollama") from e
            model_name = kwargs.get("model_name", "phi4-mini")
            return LocalSLMOrchestrator(model_name=model_name)

        elif orchestrator_type == "claude":
            # Lazy import to avoid dependency if not using claude
            try:
                from .claude_orchestrator import ClaudeOrchestrator
            except ImportError as e:
                raise ImportError("Anthropic not installed. Run: pip install anthropic") from e
            api_key = kwargs.get("api_key")
            model = kwargs.get("model", "claude-sonnet-4-5-20250929")
            if not api_key:
                raise ValueError("Claude orchestrator requires api_key")
            return ClaudeOrchestrator(api_key=api_key, model=model)

        else:
            valid_types = ["claude", "local", "rule-based"]
            raise ValueError(
                f"Unknown orchestrator type: {orchestrator_type}. Must be one of {valid_types}"
            )
