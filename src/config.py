"""Configuration loading and validation.

Loads a YAML config into validated Pydantic models, reads the LLM API
key from the environment (failing clearly if absent), resolves the model
name from an optional environment override, and creates output folders.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatasetConfig(BaseModel):
    name: str = "nuscenes"
    dataroot: str
    version: str


class PipelineConfig(BaseModel):
    max_samples: int = 100
    sample_stride: int = 1
    target_verified_count: int = 100
    max_questions_per_sample: int = 3
    max_candidates_for_gui: int = 300


class LLMConfig(BaseModel):
    # Provider: 'nvidia' (default), 'gemini', or 'mock'.
    provider: str = "nvidia"
    # NVIDIA NIM endpoint; ignored when provider is 'gemini'.
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    model_name_env: str = "LLM_MODEL_NAME"
    default_model_name: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    max_calls_per_run: int = 50
    request_timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_seconds: float = 5.0
    min_seconds_between_calls: float = 1.0
    cache_enabled: bool = True
    cache_dir: str = "outputs/cache/llm"
    # Max tokens in the model's completion (not prompt).
    max_output_tokens: int = 4096
    use_structured_output: bool = False
    image_input_mode: str = "inline"
    max_images_per_request: int = 6
    max_image_side_pixels: int = 1280
    # When true (or env LLM_MOCK / GEMINI_MOCK is set), use the mock client.
    mock_mode: bool = False


# Legacy alias so any external code referencing GeminiConfig keeps working.
GeminiConfig = LLMConfig


class PathsConfig(BaseModel):
    sample_index_path: str = "outputs/cache/nuscenes_sample_index.jsonl"
    preview_dir: str = "outputs/cache/previews"
    auto_candidates_path: str = "outputs/candidates/auto_candidates.jsonl"
    verified_output_path: str = "outputs/verified/ground_truth_100.jsonl"
    rejected_output_path: str = "outputs/verified/rejected_samples.jsonl"
    report_dir: str = "outputs/reports"


class GuiConfig(BaseModel):
    default_camera: str = "CAM_FRONT"
    show_multiview_grid: bool = True


class AppConfig(BaseModel):
    dataset: DatasetConfig
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    gui: GuiConfig = Field(default_factory=GuiConfig)

    @property
    def gemini(self) -> LLMConfig:
        """Back-compat alias: ``config.gemini`` maps to ``config.llm``."""
        return self.llm


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML config file, then create output folders.

    Also loads a local ``.env`` file (if present) so the Gemini API key
    can be supplied without exporting it manually.
    """
    load_dotenv()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = AppConfig.model_validate(raw)
    ensure_output_directories(config)
    return config


def ensure_output_directories(config: AppConfig) -> None:
    """Create every output directory referenced by the config."""
    directories = [
        config.llm.cache_dir,
        config.paths.preview_dir,
        config.paths.report_dir,
        Path(config.paths.sample_index_path).parent,
        Path(config.paths.auto_candidates_path).parent,
        Path(config.paths.verified_output_path).parent,
        Path(config.paths.rejected_output_path).parent,
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def resolve_model_name(config: AppConfig) -> str:
    """Return the LLM model name, preferring the env override."""
    env_value = os.environ.get(config.llm.model_name_env)
    if env_value:
        return env_value
    return config.llm.default_model_name


def load_required_env(var_name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"Required environment variable '{var_name}' is not set. "
            f"Set it (e.g. in a .env file) before running this stage."
        )
    return value


def get_llm_api_key(config: AppConfig) -> str:
    """Return the LLM API key from the configured environment variable."""
    return load_required_env(config.llm.api_key_env)


# Legacy alias.
get_gemini_api_key = get_llm_api_key


def get_optional_env(var_name: str) -> Optional[str]:
    """Return an optional environment variable, or None if unset."""
    return os.environ.get(var_name)
