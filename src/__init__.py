"""PyTorch LLM Phase 0 - Training pipeline."""

from src.config import Config, parse_cli_args, load_config
from src.checkpoint import CheckpointManager
from src.data import DummyDataset, create_dataloaders
from src.logger import setup_logger, MetricsLogger
from src.model import SimpleMLP, create_model, count_parameters
from src.run_manager import RunManager
from src.system import get_system_info, SystemInfo
from src.train_utils import (
    train_epoch,
    evaluate,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "parse_cli_args",
    "load_config",
    "CheckpointManager",
    "DummyDataset",
    "create_dataloaders",
    "setup_logger",
    "MetricsLogger",
    "SimpleMLP",
    "create_model",
    "count_parameters",
    "RunManager",
    "get_system_info",
    "SystemInfo",
    "train_epoch",
    "evaluate",
    "get_linear_schedule_with_warmup",
    "get_cosine_schedule_with_warmup",
]
