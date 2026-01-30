# PyTorch LLM Phase 0 - GitHub Copilot Instructions

## Project Overview

**pytorch-llm-phase0** is a production-ready PyTorch training framework for Apple Silicon (M1/M2) Macs.

**Purpose**: Establish enterprise-grade ML development standards with emphasis on reproducibility, device optimization (MPS), and code quality.

**Phase**: Phase 0 (Hito 0) - Foundational infrastructure for reproducible training pipelines on M1.

---

## Code Quality Standards

### Mandatory Quality Gates

ALL code must pass these checks before commit:

```bash
make lint    # ruff - Fast Python linter
make typecheck  # mypy - Static type checking
make format  # black - Code formatting (line length: 100)
make test    # pytest - Unit tests
```

#### Linting (ruff)

- **No unused imports** - Remove all `import X` that aren't used
- **No undefined names** - All variables must be defined
- **Import ordering**:
  1. Standard library (e.g., `import os`, `import json`)
  2. Third-party (e.g., `import torch`, `import numpy`)
  3. Local imports (e.g., `from src.config import Config`)
- **No trailing whitespace** - Clean line endings

#### Type Checking (mypy)

- **All function signatures must have type hints**:
  ```python
  def train_epoch(
      model: nn.Module,
      dataloader: DataLoader,
      device: str,
  ) -> Dict[str, float]:
  ```
- **Dict/List type hints are required**:
  ```python
  # GOOD
  results: Dict[str, float] = {}
  items: List[int] = []
  
  # BAD - no type hints
  results = {}
  items = []
  ```
- **Optional types must be explicit**:
  ```python
  # GOOD
  scheduler: Optional[LRScheduler] = None
  
  # BAD
  scheduler = None  # mypy will complain
  ```
- **Use `# type: ignore` sparingly** - Only for unavoidable third-party library incompatibilities

#### Code Formatting (black)

- **Line length**: 100 characters (NOT 88, that's default)
- **Quotes**: Use double quotes `"` (black standard)
- **Spacing**: Black auto-formats, just run `make format`

### Running Quality Checks

**Before every commit**:
```bash
# Check all
make all

# Or individually
make lint      # Check linting
make typecheck # Check types
make format    # Auto-format code
make test      # Run tests (currently minimal)
```

**Fix issues**:
```bash
ruff check src/ --fix  # Auto-fix linting issues
black src/ tests/      # Auto-format
# mypy: must fix manually (it's type-checking, not auto-fixable)
```

---

## Project Structure

```
pytorch-llm-phase0/
├── src/                    # Main codebase
│   ├── __init__.py        # Public API exports
│   ├── config.py          # Configuration management (YAML → CLI override)
│   ├── train.py           # Training entry point
│   ├── train_utils.py     # Training loop, schedulers, evaluation
│   ├── model.py           # SimpleMLP model architecture
│   ├── data.py            # DummyDataset, DataLoader creation
│   ├── checkpoint.py      # Save/load checkpoints (model + optimizer + metadata)
│   ├── logger.py          # Structured logging (JSON + rich console)
│   ├── run_manager.py     # Run directory management (runs/YYYYMMDD_HHMMSS_tag/)
│   ├── system.py          # System info collection (Python, torch, git, device)
│   ├── device.py          # MPS/CPU device management with fallback
│   └── benchmark.py       # Performance benchmarking (MPS vs CPU)
├── configs/
│   └── train.yaml         # Training config (seed, batch_size, lr, device, precision)
├── scripts/
│   └── benchmark.py       # Executable: `python scripts/benchmark.py`
├── tests/                 # Unit tests (pytest)
├── docs/
│   └── PHASE_E_MPS_GUIDE.md  # Comprehensive MPS optimization guide
├── Makefile               # Quality gates & task automation
├── requirements.txt       # Python dependencies (uv pip freeze)
└── .github/
    └── copilot-instructions.md  # This file
```

---

## Key Modules and APIs

### Configuration (`src/config.py`)

**Purpose**: Load training config from YAML with CLI overrides.

```python
from src import parse_cli_args, load_config

args = parse_cli_args()  # --config, --lr, --batch-size, --epochs, etc.
config = load_config(args)
print(config.training.learning_rate)  # 1e-3 (from YAML) or CLI override
```

**Config structure**:
```python
config.seed                    # int: reproducibility seed
config.device                  # str: "mps" or "cpu"
config.precision              # str: "fp32", "fp16", "bf16"
config.training.epochs        # int
config.training.batch_size    # int
config.training.learning_rate # float
config.model.hidden_size      # int: MLP hidden dimension
config.logging.level          # str: "INFO", "DEBUG"
config.logging.save_interval  # int: log every N steps
```

### Training Loop (`src/train_utils.py`)

**Purpose**: Forward/backward/step with gradient clipping and metrics.

```python
from src import train_epoch, evaluate

# Train one epoch
metrics = train_epoch(
    model=model,
    dataloader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=loss_fn,
    device="mps",
    epoch=0,
    run_manager=run_manager,
)
# Returns: {'loss': float, 'accuracy': float, 'time_seconds': float}

# Validate (no gradients)
val_metrics = evaluate(model, val_loader, criterion, device="mps")
# Returns: {'loss': float, 'accuracy': float}
```

### Checkpointing (`src/checkpoint.py`)

**Purpose**: Save/load full training state (model + optimizer + metadata).

```python
from src import CheckpointManager

ckpt_manager = CheckpointManager(checkpoint_dir=Path("checkpoints"))

# Save checkpoint
ckpt_manager.save(
    epoch=5,
    step=1000,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    metrics={"val_loss": 0.5},
    is_best=True,  # Also saves to "best.pt"
)

# Resume from checkpoint
info = ckpt_manager.load(
    checkpoint_path=Path("checkpoints/checkpoint_epoch_005.pt"),
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
)
start_epoch = info["epoch"] + 1
```

### Device Management (`src/device.py`)

**Purpose**: MPS availability check with CPU fallback and dtype support.

```python
from src import is_mps_available, get_device, get_dtype_from_string

# Smart device selection
device = get_device(preferred_device="mps", fallback_on_error=True)
# Returns "mps" if available, else "cpu"

# Check explicitly
if is_mps_available():
    print("✓ MPS available")

# Dtype support
dtype = get_dtype_from_string("fp32")  # torch.float32
dtype = get_dtype_from_string("fp16")  # torch.float16
```

### Benchmarking (`src/benchmark.py`)

**Purpose**: Compare MPS vs CPU performance.

```python
from src import Benchmark

# Run benchmark
results = Benchmark.compare_devices(
    model_fn=lambda: create_model(),
    dataloader=train_loader,
    criterion=loss_fn,
    optimizer_fn=lambda m: torch.optim.AdamW(m.parameters()),
    devices=["mps", "cpu"],
    dtypes=["fp32"],
    num_steps=50,
    warmup_steps=10,
)

Benchmark.print_comparison(results)
# Shows: throughput (samples/s), avg time (ms), median, min, max
```

### Run Manager (`src/run_manager.py`)

**Purpose**: Create timestamped run directories with auditable logs.

```python
from src import RunManager

run_manager = RunManager(config=config, tag="experiment1")
# Creates: runs/20260129_220634_experiment1/

# Log metrics
run_manager.log_metrics(
    metrics={"loss": 0.5, "acc": 0.92},
    step=100,
)

# Access paths
checkpoint_dir = run_manager.get_checkpoint_dir()
# Returns: runs/20260129_220634_experiment1/checkpoints/

# Saved artifacts
# - config_resolved.yaml: Final config used
# - system.json: Python, torch, git, device info
# - metrics.jsonl: One metric dict per line
# - train.log: Structured logs
```

---

## Development Workflow

### 1. **Before Starting Work**

```bash
# Activate environment
source .venv/bin/activate

# Update dependencies if needed
uv pip install -r requirements.txt
```

### 2. **Write Code**

Follow these patterns:

#### New Module/Function

```python
"""Module docstring - what this does."""

import math  # stdlib
import time
from typing import Dict, Optional

import torch  # third-party
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config  # local imports


def my_function(
    x: torch.Tensor,
    y: int,
    device: str = "cpu",
) -> Dict[str, float]:
    """Function docstring with args, returns, and examples.
    
    Args:
        x: Input tensor
        y: Integer parameter
        device: Device to use
        
    Returns:
        Dictionary with results
    """
    result = {"metric": 1.0}
    return result
```

#### Type Hints (Required)

```python
# GOOD - Full type hints
def train(
    model: nn.Module,
    dataloader: DataLoader,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    return metrics

# BAD - Missing types (will fail mypy)
def train(model, dataloader):  # ❌
    metrics = {}
    return metrics
```

### 3. **Run Quality Checks**

```bash
# All checks
make all

# Fix automatically
ruff check src/ --fix && black src/ tests/

# Fix types manually (mypy won't auto-fix)
# Read error message, edit code, run again
make typecheck
```

### 4. **Commit**

```bash
git add .
git commit -m "Brief title

Detailed explanation of changes:
- What was added/changed
- Why (motivation)
- Any important notes"
```

**Commit message guidelines**:
- First line: imperative mood, under 50 chars
- Blank line
- Detailed explanation (wrap at ~72 chars)
- Reference issues/PRs if applicable

### 5. **Push**

```bash
# Create feature branch
git checkout -b ketcx/feature-name

# Push to GitHub
git push origin ketcx/feature-name

# Open PR on GitHub
# Add description, link issues, request review
```

---

## Mac M1 / MPS Best Practices

### Device Selection

```python
from src import get_device, is_mps_available

# In training code
device = get_device(preferred_device="mps", fallback_on_error=True)

# On larger models/batches, MPS on M1 can be significantly faster than CPU
# For small models, CPU may be faster (see README benchmarks)
# Always benchmark with `python scripts/benchmark.py` and choose based on results
```

### Data Types

```python
# Recommended for M1
precision = "fp32"  # Stable, safest choice

# Only use if benchmarking shows benefit
precision = "fp16"  # Faster but less stable

# Avoid on M1
precision = "bf16"  # Limited MPS support
```

### Common OOM Solutions

1. **Reduce batch size** (most effective):
   ```python
   batch_size = 32  # Start conservative, increase gradually
   ```

2. **Use no_grad() for inference**:
   ```python
   with torch.no_grad():
       output = model(batch)  # No gradient overhead
   ```

3. **Clear gradients explicitly**:
   ```python
   optimizer.zero_grad()  # Before backward()
   loss.backward()
   optimizer.step()
   ```

4. **Don't accumulate tensors**:
   ```python
   # BAD
   outputs = []
   for batch in loader:
       outputs.append(model(batch))  # Accumulates in memory!
   
   # GOOD
   total_loss = 0.0
   for batch in loader:
       loss = criterion(model(batch), target)
       total_loss += loss.item()  # Keep only scalar
   ```

### Benchmarking Your Changes

```bash
# Before optimization
python scripts/benchmark.py
# Baseline: MPS 5350.98 samples/s

# After optimization
python scripts/benchmark.py
# New: MPS 6200.50 samples/s
# Improvement: 15.8%
```

---

## Testing

### Run Tests

```bash
make test  # Run all tests with pytest
```

### Write Tests

```python
# tests/test_model.py
import pytest
import torch
from src.model import create_model

def test_model_forward():
    """Test model forward pass."""
    model = create_model(input_dim=10, hidden_size=128)
    batch = torch.randn(32, 10)
    
    output = model(batch)
    
    assert output.shape == (32, 2)  # batch_size, num_classes
    
def test_model_device():
    """Test model can move to device."""
    model = create_model()
    model.to("cpu")
    assert next(model.parameters()).device.type == "cpu"
```

---

## Documentation

### Important Files

- **[PHASE_E_MPS_GUIDE.md](docs/PHASE_E_MPS_GUIDE.md)** - MPS optimization, dtypes, OOM patterns
- **[README.md](README.md)** - Project overview, installation, usage
- **[Makefile](Makefile)** - Available commands and targets

### Add Docstrings

```python
def my_function(x: int) -> str:
    """One-line summary of what this does.
    
    Longer description if needed, explaining the purpose and any
    important behavior.
    
    Args:
        x: Description of parameter x
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
        
    Example:
        >>> result = my_function(42)
        >>> print(result)
        "42"
    """
    return str(x)
```

---

## Common Issues & Solutions

### Issue: "mypy error: No library stubs"

**Solution**: Install type stubs
```bash
uv pip install types-PyYAML
# or
mypy --install-types
```

### Issue: "ruff: unused import"

**Solution**: Remove or use the import
```python
# BAD
import json  # But never used json

# GOOD
# (Just delete the line)
```

### Issue: "MPS out of memory"

**Solution**: 
1. Reduce batch_size in config
2. Check for tensor accumulation (see OOM Solutions above)
3. Fall back to CPU: `device = "cpu"`

### Issue: "Test failed after my changes"

**Solution**:
```bash
# Run specific test
pytest tests/test_module.py::test_function -v

# Run with print statements
pytest tests/test_module.py -v -s
```

---

## Performance Profiling

### Check Training Speed

```python
# Built into train loop - look at train.log
# Or use timestamps
import time
start = time.perf_counter()
# ... training step ...
elapsed = time.perf_counter() - start
print(f"Step time: {elapsed*1000:.2f}ms")
```

### Profile with torch.profiler

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU],
    record_shapes=True,
) as prof:
    output = model(batch)
    loss = criterion(output, target)

prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
```

---

## Version Control

### Branch Naming

```bash
git checkout -b ketcx/feature-description
# Examples:
# ketcx/add-amp-support
# ketcx/fix-oom-issue
# ketcx/optimize-dataloader
```

### Commit Best Practices

- **One logical change per commit**
- **Commit only passing code** (make all must pass)
- **Write clear messages** (first line: title, then details)
- **Don't commit sensitive data** (already in .gitignore)

---

## Useful Commands

```bash
# Training
python -m src.train --config configs/train.yaml --epochs 5 --lr 2e-3

# Resume from checkpoint
python -m src.train --config configs/train.yaml --resume runs/20260129_220634_exp1/checkpoints/best.pt

# Benchmark
python scripts/benchmark.py

# View run artifacts
ls runs/20260129_220634_*/
cat runs/20260129_220634_*/config_resolved.yaml
cat runs/20260129_220634_*/system.json

# Check quality
make lint && make typecheck && make format

# Update dependencies
uv pip freeze > requirements.txt
git add requirements.txt && git commit -m "Update dependencies"
```

---

## References

- **PyTorch Docs**: https://pytorch.org/docs/stable/
- **MPS Guide**: [PHASE_E_MPS_GUIDE.md](docs/PHASE_E_MPS_GUIDE.md)
- **ruff**: https://docs.astral.sh/ruff/
- **mypy**: https://mypy.readthedocs.io/
- **black**: https://black.readthedocs.io/
