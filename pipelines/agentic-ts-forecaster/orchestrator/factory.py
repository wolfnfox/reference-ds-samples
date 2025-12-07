from typing import Any

from config import DEFAULT_CLAUDE_MODEL, DEFAULT_LOCAL_MODEL, OrchestratorType

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
        if orchestrator_type == OrchestratorType.RULE_BASED:
            return RuleBasedOrchestrator()

        elif orchestrator_type == OrchestratorType.LOCAL:
            # Lazy import to avoid dependency if not using local
            try:
                from .local_orchestrator import LocalSLMOrchestrator
            except ImportError as e:
                missing_module = getattr(e, 'name', None)
                if missing_module == 'ollama':
                    raise ImportError("Ollama not installed. Run: pip install ollama") from e
                else:
                    raise
            model_name = kwargs.get("model_name", DEFAULT_LOCAL_MODEL)
            return LocalSLMOrchestrator(model_name=model_name)

        elif orchestrator_type == OrchestratorType.CLAUDE:
            # Lazy import to avoid dependency if not using claude
            try:
                from .claude_orchestrator import ClaudeOrchestrator
            except ImportError as e:
                missing_module = getattr(e, 'name', None)
                if missing_module == 'anthropic':
                    raise ImportError("Anthropic not installed. Run: pip install anthropic") from e
                else:
                    raise
            api_key = kwargs.get("api_key")
            model = kwargs.get("model", DEFAULT_CLAUDE_MODEL)
            if not api_key:
                raise ValueError("Claude orchestrator requires api_key")
            return ClaudeOrchestrator(api_key=api_key, model=model)

        else:
            valid_types = list(OrchestratorType)
            raise ValueError(
                f"Unknown orchestrator type: {orchestrator_type}. Must be one of {valid_types}"
            )
