"""PyTorch LLM Phase 0 - Training pipeline."""

from src.benchmark import Benchmark
from src.checkpoint import CheckpointManager
from src.config import Config, load_config, parse_cli_args
from src.data import DummyDataset, create_dataloaders
from src.device import (
    DeviceContext,
    check_mps_limitations,
    get_device,
    get_device_info,
    get_dtype_from_string,
    is_mps_available,
    is_mps_supported,
    print_device_stats,
)
from src.logger import MetricsLogger, setup_logger
from src.model import SimpleMLP, count_parameters, create_model
from src.run_manager import RunManager
from src.system import SystemInfo, get_system_info
from src.train_utils import (
    evaluate,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    train_epoch,
)

__version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "CheckpointManager",
    "Config",
    "DummyDataset",
    "DeviceContext",
    "MetricsLogger",
    "RunManager",
    "SimpleMLP",
    "SystemInfo",
    "check_mps_limitations",
    "count_parameters",
    "create_dataloaders",
    "create_model",
    "evaluate",
    "get_cosine_schedule_with_warmup",
    "get_device",
    "get_device_info",
    "get_dtype_from_string",
    "get_linear_schedule_with_warmup",
    "get_system_info",
    "is_mps_available",
    "is_mps_supported",
    "load_config",
    "parse_cli_args",
    "print_device_stats",
    "setup_logger",
    "train_epoch",
]
