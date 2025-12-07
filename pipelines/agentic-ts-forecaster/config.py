from dataclasses import dataclass, field
from dotenv import load_dotenv
from enum import StrEnum
from typing import Any, List, Optional
import os
import re
import yaml


class OrchestratorType(StrEnum):
    CLAUDE = "claude"
    LOCAL = "local"
    RULE_BASED = "rule-based"


# Default model names
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_LOCAL_MODEL = "phi4-mini"

# Default pipeline settings
DEFAULT_TARGET_COLUMN = "target"
DEFAULT_DATE_COLUMN = "date"
DEFAULT_HORIZON = 12
DEFAULT_TEST_SIZE = 0.2
DEFAULT_OUTPUT_DIR = "./outputs"


def _expand_env_vars(value: Any) -> Any:
    """Expand ${VAR} patterns in strings with environment variable values."""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")
        matches = pattern.findall(value)
        for var_name in matches:
            env_val = os.environ.get(var_name, "")
            value = value.replace(f"${{{var_name}}}", env_val)
        return value
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator selection and settings."""

    type: str = OrchestratorType.RULE_BASED
    claude_api_key: Optional[str] = None
    claude_model: str = DEFAULT_CLAUDE_MODEL
    local_model_name: str = DEFAULT_LOCAL_MODEL

    def __post_init__(self):
        valid_types = set(OrchestratorType)
        if self.type not in valid_types:
            raise ValueError(f"Invalid orchestrator type: {self.type}. Must be one of {valid_types}")

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        """Load configuration from environment variables."""
        load_dotenv()
        return cls(
            type=os.environ.get("ORCHESTRATOR_TYPE", OrchestratorType.RULE_BASED),
            claude_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            claude_model=os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
            local_model_name=os.environ.get("LOCAL_MODEL", DEFAULT_LOCAL_MODEL),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "OrchestratorConfig":
        """Load configuration from YAML file."""
        load_dotenv()

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML configuration file not found: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        
        orch_data = data.get("orchestrator", {})
        orch_data = _expand_env_vars(orch_data)
        return cls(
            type=orch_data.get("type", OrchestratorType.RULE_BASED),
            claude_api_key=orch_data.get("claude_api_key"),
            claude_model=orch_data.get("claude_model", DEFAULT_CLAUDE_MODEL),
            local_model_name=orch_data.get("local_model_name", DEFAULT_LOCAL_MODEL),
        )


@dataclass
class PipelineConfig:
    """Configuration for the forecasting pipeline."""

    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    target_column: str = DEFAULT_TARGET_COLUMN
    date_column: str = DEFAULT_DATE_COLUMN
    exogenous_columns: List[str] = field(default_factory=list)
    horizon: int = DEFAULT_HORIZON
    test_size: float = DEFAULT_TEST_SIZE
    force_models: List[str] = field(default_factory=list)
    enable_market_research: bool = False
    output_dir: str = DEFAULT_OUTPUT_DIR
    generate_report: bool = True
    generate_presentation: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load full pipeline configuration from YAML file."""
        load_dotenv()
        
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML configuration file not found: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        
        data = _expand_env_vars(data)

        orch_data = data.get("orchestrator", {})
        orchestrator = OrchestratorConfig(
            type=orch_data.get("type", OrchestratorType.RULE_BASED),
            claude_api_key=orch_data.get("claude_api_key"),
            claude_model=orch_data.get("claude_model", DEFAULT_CLAUDE_MODEL),
            local_model_name=orch_data.get("local_model_name", DEFAULT_LOCAL_MODEL),
        )

        pipeline_data = data.get("pipeline", {})
        return cls(
            orchestrator=orchestrator,
            target_column=pipeline_data.get("target_column", DEFAULT_TARGET_COLUMN),
            date_column=pipeline_data.get("date_column", DEFAULT_DATE_COLUMN),
            exogenous_columns=pipeline_data.get("exogenous_columns", []),
            horizon=pipeline_data.get("horizon", DEFAULT_HORIZON),
            test_size=pipeline_data.get("test_size", DEFAULT_TEST_SIZE),
            force_models=pipeline_data.get("force_models", []),
            enable_market_research=pipeline_data.get("enable_market_research", False),
            output_dir=pipeline_data.get("output_dir", DEFAULT_OUTPUT_DIR),
            generate_report=pipeline_data.get("generate_report", True),
            generate_presentation=pipeline_data.get("generate_presentation", False),
        )
