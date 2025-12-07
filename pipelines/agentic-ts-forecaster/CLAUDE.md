# Claude Code Development Tasks for Time Series Forecasting Agent

## Project Context

You are building an agentic time series forecasting system with:
- Pluggable orchestrators (Claude API, local SLMs via Ollama, rule-based)
- Model-agnostic architecture supporting multiple time series models
- Automated pipeline from data ingestion to report generation

**Current Status**: Basic project structure and base abstractions are in place.

**Package Management**: This project uses `uv` for fast, reliable Python package management. Always use `uv` commands instead of `pip`.

## Project Structure

The project has the following directory structure:
````
forecast-agent/
├── cli.py
├── pipeline.py
├── config.py
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── config/
│   ├── claude_online.yaml
│   ├── local_offline.yaml
│   └── rule_based.yaml
├── orchestrator/
│   ├── __init__.py
│   ├── base.py
│   ├── claude_orchestrator.py
│   ├── local_orchestrator.py
│   ├── rule_based_orchestrator.py
│   └── factory.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── patchtsmixer.py
│   ├── arima.py
│   ├── prophet.py
│   └── lstm.py
├── agents/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── eda.py
│   ├── training.py
│   └── reporting.py
├── tests/
│   ├── __init__.py
│   ├── test_orchestrators.py
│   └── test_models.py
└── outputs/
````

## Development Phases

---

## PHASE 1: Rule-Based Orchestrator & Configuration System

**Goal**: Get a working end-to-end pipeline with deterministic orchestrator (no LLM dependencies).

### Task 1.1: Implement Configuration System

**File**: `config.py`

**Requirements**:
- Create `OrchestratorConfig` dataclass with fields:
  - `type: str` (claude/local/rule-based)
  - `claude_api_key: Optional[str]`
  - `claude_model: str` (default: "claude-sonnet-4-5-20250929")
  - `local_model_name: str` (default: "phi4-mini")
- Add class methods: `from_env()` and `from_yaml()`
- Create `PipelineConfig` dataclass with fields:
  - `orchestrator: OrchestratorConfig`
  - `target_column: str`
  - `date_column: str`
  - `exogenous_columns: List[str]`
  - `horizon: int`
  - `test_size: float`
  - `force_models: List[str]`
  - `enable_market_research: bool`
  - `output_dir: str`
  - `generate_report: bool`
  - `generate_presentation: bool`
- Add class method: `from_yaml()`

**Dependencies**: `pyyaml`, `python-dotenv`

---

### Task 1.2: Implement Rule-Based Orchestrator

**File**: `orchestrator/rule_based_orchestrator.py`

**Requirements**:
- Implement `RuleBasedOrchestrator` class inheriting from `OrchestratorBase`
- Implement `select_models()` method with these rules:
  - Always include ARIMA as baseline (if available)
  - For n_samples > 200: add PatchTSMixer (if multivariate) or LSTM
  - For high seasonality: add Prophet
  - Limit to max 4 models
  - Return `ModelSelectionDecision` with rule-based rationale
- Implement `analyze_results()` method:
  - Sort models by RMSE
  - Generate simple insights comparing performance
  - Return dict with best_model, insights, confidence_level
- Implement `generate_report()` method:
  - Generate markdown formatted report
  - Include sections: Best Model, Model Comparison, Business Implications, Considerations

**No external dependencies** (pure Python logic)

---

### Task 1.3: Implement Orchestrator Factory

**File**: `orchestrator/factory.py`

**Requirements**:
- Create `OrchestratorFactory` class with static method `create()`
- Accept `orchestrator_type` parameter ('claude', 'local', 'rule-based')
- Accept `**kwargs` for type-specific configuration
- Return appropriate orchestrator instance
- Raise `ValueError` for unknown orchestrator types

**Import**: `OrchestratorBase`, `RuleBasedOrchestrator` (and later Claude/Local)

---

### Task 1.4: Create YAML Configuration Files

**Files**: 
- `config/rule_based.yaml`
- `config/local_offline.yaml` 
- `config/claude_online.yaml`

**Requirements**:
- Each file should specify appropriate orchestrator settings
- Include common pipeline settings (target_column, horizon, etc.)
- Use environment variable substitution for API keys: `${ANTHROPIC_API_KEY}`
- See README.md for example structure

---

## PHASE 2: Model Registry & Data Agents

**Goal**: Implement model management and data processing infrastructure.

### Task 2.1: Implement Model Registry

**File**: `models/registry.py`

**Requirements**:
- Create `ModelRegistry` class
- Implement `__init__()` with empty `_models` dict, call `_register_defaults()`
- Implement `register(model: TimeSeriesModelAgent)` - add to `_models`
- Implement `get_model(name: str) -> TimeSeriesModelAgent`
- Implement `list_models() -> List[str]`
- Implement `get_compatible_models(data_characteristics: Dict) -> List[str]`:
  - Filter models based on:
    - `is_multivariate` vs `handles_multivariate`
    - `has_exogenous` vs `handles_exogenous`
    - `n_samples` vs `min_samples`
    - `n_features` vs `max_features`
  - Return list of compatible model names

**Note**: `_register_defaults()` will be empty initially, models added in Phase 3.

---

### Task 2.2: Implement Data Ingestion Agent

**File**: `agents/data_ingestion.py`

**Requirements**:
- Create `DataIngestionAgent` class
- Implement `load_and_validate(excel_path: str) -> pd.DataFrame`:
  - Load Excel file using pandas
  - Basic validation (not empty, has columns)
  - Handle common errors gracefully
  - Return cleaned DataFrame
- Implement `generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]`:
  - Calculate missing value percentages
  - Identify column types
  - Detect outliers (simple IQR method)
  - Return dict with quality metrics

**Dependencies**: `pandas`, `numpy`

---

### Task 2.3: Implement EDA Agent

**File**: `agents/eda.py`

**Requirements**:
- Create `EDAAgent` class
- Implement `profile_data(df: pd.DataFrame, target_column: str, date_column: str) -> Dict[str, Any]`:
  - Calculate: n_samples, n_features, is_multivariate, has_exogenous
  - Detect seasonality (use simple autocorrelation check)
  - Detect trend (use simple linear regression slope)
  - Calculate missing data percentage
  - Return comprehensive data profile dict
- Implement `detect_seasonality(series: pd.Series) -> bool`:
  - Use autocorrelation function (ACF) with pandas
  - Return True if significant ACF at seasonal lags (12, 24, etc.)
- Implement `detect_trend(series: pd.Series) -> bool`:
  - Fit simple linear regression
  - Return True if slope is statistically significant

**Dependencies**: `pandas`, `numpy`, `scipy` (for stats)

---

## PHASE 3: Baseline Time Series Models

**Goal**: Implement 2-3 baseline models to test the pipeline.

### Task 3.1: Implement ARIMA Model Agent

**File**: `models/arima.py`

**Requirements**:
- Create `ARIMAAgent` class inheriting from `TimeSeriesModelAgent`
- Implement all abstract methods from base class
- Properties:
  - `name`: return "ARIMA"
  - `capabilities`: ModelCapabilities with appropriate settings
- Methods:
  - `train()`: Use statsmodels SARIMAX, fit model, store as instance variable
  - `predict()`: Generate forecast for given horizon
  - `evaluate()`: Calculate MAE, RMSE, R², MAPE
  - `interpret()`: Return dict with model coefficients, AIC, BIC
- Use SARIMAX to support seasonal and exogenous variables
- Handle errors gracefully (model convergence issues)

**Dependencies**: `statsmodels`, `pandas`, `numpy`, `scikit-learn`

---

### Task 3.2: Implement Prophet Model Agent

**File**: `models/prophet.py`

**Requirements**:
- Create `ProphetAgent` class inheriting from `TimeSeriesModelAgent`
- Implement all abstract methods from base class
- Properties:
  - `name`: return "Prophet"
  - `capabilities`: ModelCapabilities with appropriate settings
- Methods:
  - `train()`: Initialize Prophet model, add regressors if exogenous, fit
  - `predict()`: Generate forecast with uncertainty intervals
  - `evaluate()`: Calculate standard metrics
  - `interpret()`: Return component analysis (trend, seasonality)
- Handle Prophet's specific data format requirements (ds, y columns)
- Suppress verbose Prophet output

**Dependencies**: `prophet`, `pandas`, `numpy`, `scikit-learn`

---

### Task 3.3: Update Model Registry with Defaults

**File**: `models/registry.py`

**Requirements**:
- Import ARIMAAgent and ProphetAgent
- In `_register_defaults()` method:
  - Instantiate and register ARIMAAgent()
  - Instantiate and register ProphetAgent()
- Ensure models are available when registry is created

---

## PHASE 4: Training Agent & Pipeline Integration

**Goal**: Build training infrastructure and wire up the complete pipeline.

### Task 4.1: Implement Training Agent

**File**: `agents/training.py`

**Requirements**:
- Create `TrainingAgent` class
- Accept `ModelRegistry` in `__init__`
- Implement `train_parallel()` method:
  - Accept: data, model_names, target_column, exogenous_columns, test_size, horizon
  - Split data into train/test
  - For each model_name:
    - Get model from registry
    - Call `run_full_pipeline()`
    - Collect ModelResult
  - Return List[ModelResult]
- Use multiprocessing or concurrent.futures for parallel execution
- Handle individual model failures gracefully (log error, continue)
- Add progress indicators using Rich

**Dependencies**: `concurrent.futures`, `rich`

---

### Task 4.2: Implement Main Pipeline

**File**: `pipeline.py`

**Requirements**:
- Create `ForecastPipeline` class
- Accept `PipelineConfig` in `__init__`:
  - Create orchestrator using OrchestratorFactory
  - Initialize ModelRegistry
  - Initialize all agent classes
  - Create output directory
- Implement `run(excel_path: str, business_context: str = "") -> dict`:
  - Step 1: Data ingestion (DataIngestionAgent)
  - Step 2: EDA & profiling (EDAAgent)
  - Step 3: Model selection (orchestrator.select_models)
  - Step 4: Parallel training (TrainingAgent)
  - Step 5: Result analysis (orchestrator.analyze_results)
  - Step 6: Market research (if enabled and Claude orchestrator)
  - Step 7: Report generation (orchestrator.generate_report)
  - Print progress for each step using Rich
  - Return dict with all results
- Handle errors at each step with informative messages

**Dependencies**: All agents, orchestrator, config

---

### Task 4.3: Update CLI to Execute Pipeline

**File**: `cli.py`

**Requirements**:
- Update `forecast()` command:
  - Load config from YAML if provided, else create from CLI args
  - Instantiate ForecastPipeline with config
  - Call pipeline.run() with excel_path and context
  - Display results summary using Rich tables
  - Show output file paths
- Add error handling and user-friendly messages
- Add `--context` option for business context string
- Add `--local-model` option for SLM model name

---

## PHASE 5: Local SLM Orchestrator

**Goal**: Add support for local small language models via Ollama.

### Task 5.1: Implement Local SLM Orchestrator

**File**: `orchestrator/local_orchestrator.py`

**Requirements**:
- Create `LocalSLMOrchestrator` class inheriting from `OrchestratorBase`
- Accept `model_name` in `__init__` (default: "phi4-mini")
- Initialize Ollama client, verify model is available (pull if not)
- Implement `select_models()`:
  - Create simplified prompt optimized for smaller models
  - Use ollama.generate() with format='json'
  - Set temperature=0.3 for deterministic output
  - Parse response into ModelSelectionDecision
- Implement `analyze_results()`:
  - Simplified prompt for model comparison
  - Extract insights from SLM response
- Implement `generate_report()`:
  - Use SLM to generate markdown report
  - Include all key sections

**Dependencies**: `ollama`, `json`

**Prompt Engineering Tips**:
- Keep prompts concise and structured
- Use clear JSON schema examples
- Lower token counts for faster inference

---

### Task 5.2: Add SLM Commands to CLI

**File**: `cli.py`

**Requirements**:
- Add `install-slm` command:
  - Accept model name as argument
  - Use ollama.pull() to download model
  - Show progress with Rich status indicator
  - Confirm successful installation
- Add `list-slms` command:
  - Call ollama.list()
  - Display available models in Rich table
  - Show name, size, modified date

---

### Task 5.3: Update Orchestrator Factory

**File**: `orchestrator/factory.py`

**Requirements**:
- Import `LocalSLMOrchestrator`
- Add 'local' case to `create()` method
- Pass `model_name` from kwargs
- Handle missing Ollama installation gracefully

---

## PHASE 6: Claude API Orchestrator

**Goal**: Add cloud-based orchestrator with web search capability.

### Task 6.1: Implement Claude Orchestrator

**File**: `orchestrator/claude_orchestrator.py`

**Requirements**:
- Create `ClaudeOrchestrator` class inheriting from `OrchestratorBase`
- Accept `api_key` and `model` in `__init__`
- Initialize Anthropic client
- Implement `select_models()`:
  - Create detailed prompt with data profile and business context
  - Request JSON response with schema
  - Use Claude Sonnet 4
  - Parse response into ModelSelectionDecision
- Implement `analyze_results()`:
  - Comprehensive prompt for business insights
  - Extract structured analysis
- Implement `generate_report()`:
  - Use Claude to generate professional markdown report
  - Include executive summary, methodology, findings, recommendations
- Implement `conduct_market_research()` (optional helper):
  - Use web_search tool for market context
  - Synthesize findings into brief

**Dependencies**: `anthropic`, `json`

**Note**: Claude can provide higher quality analysis and report generation compared to local SLMs.

---

### Task 6.2: Update Orchestrator Factory

**File**: `orchestrator/factory.py`

**Requirements**:
- Import `ClaudeOrchestrator`
- Add 'claude' case to `create()` method
- Pass `api_key` and `model` from kwargs
- Handle missing API key gracefully

---

## PHASE 7: Reporting & Presentation Agents

**Goal**: Add professional output generation.

### Task 7.1: Implement Reporting Agent

**File**: `agents/reporting.py`

**Requirements**:
- Create `ReportAgent` class
- Implement `create_markdown_report()`:
  - Accept analysis dict, model_results, market_research
  - Generate comprehensive markdown report
  - Include visualizations (matplotlib charts saved as images)
  - Save to output directory
- Implement `create_presentation()`:
  - Accept same inputs as markdown report
  - Use python-pptx to create PowerPoint
  - Slides: Title, Executive Summary, Model Comparison, Best Model Details, Forecasts, Recommendations
  - Include charts and tables
  - Save to output directory
- Implement helper methods for chart generation:
  - `create_forecast_chart()`
  - `create_model_comparison_chart()`
  - `create_metrics_table()`

**Dependencies**: `python-pptx`, `matplotlib`, `seaborn`, `plotly`

---

### Task 7.2: Integrate Reporting into Pipeline

**File**: `pipeline.py`

**Requirements**:
- Initialize ReportAgent in `__init__`
- After Step 7 (generate_report), add Step 8:
  - If `config.generate_presentation`:
    - Call `report_agent.create_presentation()`
    - Print presentation path
- Save markdown report from orchestrator to outputs directory

---

## PHASE 8: PatchTSMixer Integration

**Goal**: Port existing PatchTSMixer implementation into the framework.

### Task 8.1: Implement PatchTSMixer Model Agent

**File**: `models/patchtsmixer.py`

**Requirements**:
- Create `PatchTSMixerAgent` class inheriting from `TimeSeriesModelAgent`
- Port your existing PatchTSMixer notebook code:
  - Model architecture
  - Data preparation (windowing, scaling)
  - Training loop
  - Prediction logic
- Implement all abstract methods:
  - `name`: return "PatchTSMixer"
  - `capabilities`: handles_multivariate=True, handles_exogenous=True, etc.
  - `train()`: Build and train PatchTSMixer model
  - `predict()`: Generate forecasts
  - `evaluate()`: Standard metrics
  - `interpret()`: SHAP analysis
- Accept hyperparameters in config dict:
  - context_length, patch_length, d_model, num_layers, etc.
- Save/load model checkpoints

**Dependencies**: `tensorflow`, `keras`, `shap`

**Reference**: Your existing PatchTSMixer notebooks

---

### Task 8.2: Register PatchTSMixer in Registry

**File**: `models/registry.py`

**Requirements**:
- Import PatchTSMixerAgent
- Add to `_register_defaults()` method

---

## PHASE 9: Testing & Documentation

**Goal**: Ensure reliability and usability.

### Task 9.1: Write Unit Tests

**Files**: `tests/test_*.py`

**Requirements**:
- Test orchestrators:
  - Model selection logic
  - Result analysis
  - Report generation
- Test model registry:
  - Registration
  - Retrieval
  - Compatibility filtering
- Test models (use synthetic data):
  - Training
  - Prediction
  - Evaluation
- Test agents:
  - Data ingestion
  - EDA profiling
  - Training coordination

**Dependencies**: `pytest`, `pytest-cov`

**Run tests**:
```bash
uv run pytest tests/ --cov
```

---

### Task 9.2: Create Example Data

**Directory**: `examples/`

**Requirements**:
- Create sample Excel file with time series data
- Include: date column, target column, 2-3 exogenous features
- Add README with description
- Provide example commands to run pipeline

---

### Task 9.3: Update Documentation

**Files**: `README.md`, docstrings

**Requirements**:
- Add comprehensive examples to README
- Document each orchestrator type with use cases
- Add troubleshooting section
- Document configuration options
- Add docstrings to all classes and methods
- Create architecture diagram (ASCII art or mermaid)
- **Include uv setup instructions**
- **Add uv command reference**

---

## PHASE 10: Future Improvements

**Goal**: Refinements and extensibility improvements.

### Task 10.1: Make Config LLM-Agnostic

**File**: `config.py`

**Requirements**:
- Refactor `OrchestratorConfig` to use generic fields:
  - `api_key: Optional[str]` instead of `claude_api_key`
  - `model_name: str` instead of `claude_model`/`local_model_name`
  - `provider: str` (anthropic/ollama/openai/etc)
- Support multiple LLM providers (OpenAI, Gemini, etc)
- Update env var names: `LLM_API_KEY`, `LLM_MODEL`, `LLM_PROVIDER`
- Maintain backwards compat with existing YAML configs

---

## Testing Checklist

After completing each phase, test with:
```bash
# Phase 1: Rule-based orchestrator
uv run python cli.py forecast examples/sample_data.xlsx --orchestrator rule-based

# Phase 5: Local SLM orchestrator
uv run python cli.py install-slm llama3.1:8b
uv run python cli.py forecast examples/sample_data.xlsx --orchestrator local

# Phase 6: Claude orchestrator
uv run python cli.py forecast examples/sample_data.xlsx --orchestrator claude --context "Test forecast"

# Use config files
uv run python cli.py forecast examples/sample_data.xlsx --config config/rule_based.yaml

# Run tests
uv run pytest tests/ --cov

# Format code
uv run black .
uv run ruff check .
```

---

## Development Tips

1. **Start Simple**: Get rule-based orchestrator working first
2. **Test Incrementally**: Test each component independently before integration
3. **Use Rich Logging**: Add informative progress indicators
4. **Handle Errors Gracefully**: Time series models can fail - catch and continue
5. **Validate Inputs**: Check data quality before passing to models
6. **Save Intermediate Results**: Helpful for debugging
7. **Use Type Hints**: Makes code more maintainable
8. **Profile Performance**: Identify bottlenecks in parallel training

---

## Priority Order

**Must Have** (Core functionality):
- Phase 1: Rule-based orchestrator
- Phase 2: Model registry & data agents  
- Phase 3: Baseline models (ARIMA, Prophet)
- Phase 4: Training agent & pipeline

**Should Have** (Enhanced capabilities):
- Phase 5: Local SLM orchestrator
- Phase 6: Claude orchestrator
- Phase 8: PatchTSMixer integration

**Nice to Have** (Polish):
- Phase 7: Presentation generation
- Phase 9: Comprehensive testing & docs
- Phase 10: Future improvements (LLM-agnostic config)

---

## Questions or Blockers?

If you encounter issues:
1. Check existing base classes for interface requirements
2. Review similar implementations (e.g., ARIMA for model structure)
3. Use Rich console for debugging output
4. Test with simple synthetic data first
5. Ask for clarification on specific implementations

Good luck with development!
