# PyTorch LLM Phase 0

**A production-ready PyTorch training pipeline with automatic reproducibility, MPS optimization, and professional tooling.**

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-4B8BBE.svg)](https://github.com/astral-sh/ruff)
[![Type: mypy](https://img.shields.io/badge/type-mypy-3776ab.svg)](https://www.mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## What is This?

A **Hito 0 (Milestone 0)** project for learning professional ML development practices with PyTorch on Apple Silicon (M1/M2/M3).

This project demonstrates:
- ✅ Clean project structure with quality gates
- ✅ Configuration management (YAML + CLI overrides)
- ✅ Reproducible training with full audit logs
- ✅ MPS (Metal Performance Shaders) optimization for M1/M2/M3
- ✅ Checkpointing and resume capabilities
- ✅ Comprehensive benchmarking tools
- ✅ Production-ready logging and metrics

**Not a target:** Training large language models. This is for **pipeline development and best practices**.

---

## Quick Start

### Prerequisites

- macOS with M1/M2/M3 (or Linux/Windows with CPU/CUDA)
- Python 3.10+
- Homebrew (optional, for uv)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ketcx/pytorch-llm-phase0.git
cd pytorch-llm-phase0

# Install uv (fast Python package manager)
brew install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Run Your First Training

```bash
# Basic training (uses defaults from configs/train.yaml)
python -m src.train --config configs/train.yaml --tag my-first-run

# Custom hyperparameters via CLI
python -m src.train \
  --config configs/train.yaml \
  --tag experiment-v1 \
  --epochs 10 \
  --batch-size 64 \
  --lr 2e-3

# Resume training from checkpoint
python -m src.train \
  --config configs/train.yaml \
  --tag resumed-experiment \
  --resume runs/20260129_225730_experiment-v1/checkpoints/best.pt
```

### 3. Check Results

```bash
# List all run directories
ls -la runs/

# View specific run
cd runs/20260129_225730_my-first-run
ls -la

# config_resolved.yaml   - Final config used
# system.json           - Hardware info (PyTorch, git, device)
# metrics.jsonl         - Metrics per epoch (JSON Lines format)
# train.log            - Full training logs
# checkpoints/          - Model checkpoints

# View metrics
cat metrics.jsonl | python -m json.tool
```

### 4. Benchmark MPS vs CPU

```bash
python scripts/benchmark.py
```

Output:
```
BENCHMARK COMPARISON
======================================================================
Config               | Avg Step (ms)   | Throughput
----------------------------------------------------------------------
cpu_fp32             |          0.44 |           84666.71 samples/s
mps_fp32             |          5.98 |            6207.14 samples/s
======================================================================

⚠️  CPU is 13.64x faster than MPS for this small model
```

---

## Configuration

### YAML Configuration

Edit [`configs/train.yaml`](configs/train.yaml):

```yaml
# Random seed for reproducibility
seed: 42

# Training hyperparameters
training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-3
  weight_decay: 1e-5

# Model architecture
model:
  hidden_size: 128
  num_layers: 2
  dropout: 0.1

# Device and precision
device: mps  # Options: mps, cpu, cuda
precision: fp32  # Options: fp32, fp16, bf16

# Paths
paths:
  runs_dir: ./runs
  checkpoints_dir: ./checkpoints
```

### CLI Overrides

Any config parameter can be overridden from CLI:

```bash
# Supported CLI flags
python -m src.train \
  --config configs/train.yaml \
  --seed 123 \
  --epochs 5 \
  --batch-size 16 \
  --lr 5e-4 \
  --device cpu \
  --tag my-experiment
```

---

## Project Structure

```
pytorch-llm-phase0/
├── src/                      # Main source code
│   ├── __init__.py
│   ├── benchmark.py         # Benchmarking suite
│   ├── checkpoint.py        # Checkpoint management
│   ├── config.py           # Configuration loading
│   ├── data.py             # Dataset and DataLoader
│   ├── device.py           # Device management (MPS/CPU)
│   ├── logger.py           # Structured logging
│   ├── model.py            # Model architecture
│   ├── run_manager.py      # Run directory management
│   ├── system.py           # System information collection
│   ├── train.py            # Training script
│   └── train_utils.py      # Training utilities
│
├── configs/                  # Configuration files
│   └── train.yaml          # Default training config
│
├── scripts/                  # Executable scripts
│   └── benchmark.py        # Benchmark script
│
├── tests/                    # Test directory
│   └── __init__.py
│
├── docs/                     # Documentation
│   └── PHASE_E_MPS_GUIDE.md # MPS optimization guide
│
├── runs/                     # Training results (gitignored)
│   └── YYYYMMDD_HHMMSS_tag/
│       ├── config_resolved.yaml
│       ├── system.json
│       ├── metrics.jsonl
│       ├── train.log
│       └── checkpoints/
│
├── checkpoints/             # Model checkpoints (gitignored)
├── Makefile                 # Quality gates commands
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── LICENSE                 # MIT License
```

---

## Training Loop

### What Happens

```python
# 1. Load configuration
config = load_config(args)

# 2. Create reproducible environment
torch.manual_seed(config.seed)
device = get_device(config.device)

# 3. Prepare data
train_loader, val_loader, test_loader = create_dataloaders(...)

# 4. Create model
model = create_model(...)
optimizer = torch.optim.AdamW(...)
scheduler = get_cosine_schedule_with_warmup(...)

# 5. Run training loop
for epoch in range(config.training.epochs):
    # Train
    metrics = train_epoch(...)
    
    # Validate
    val_metrics = evaluate(...)
    
    # Save checkpoint (and best.pt)
    checkpoint_manager.save(...)
    
    # Log metrics
    run_manager.log_metrics(...)

# 6. Evaluate on test set
test_metrics = evaluate(...)

# 7. All results saved to runs/YYYYMMDD_HHMMSS_tag/
```

### Resume Training

```bash
# Resume from best checkpoint
python -m src.train \
  --config configs/train.yaml \
  --resume runs/20260129_225730_my-run/checkpoints/best.pt \
  --tag resumed-my-run
```

The system will:
1. Load model, optimizer, scheduler state
2. Start from epoch where it left off
3. Save new results to new run directory

---

## Quality Assurance

### Run Quality Checks

```bash
# Format code
make format

# Lint code
make lint

# Type check
make typecheck

# Run tests (currently empty)
make test

# Run all checks
make all

# Clean cache files
make clean
```

### What's Checked

| Tool | Purpose | Config |
|------|---------|--------|
| **black** | Code formatting | Line length: 100 |
| **ruff** | Linting | PEP 8 compliance |
| **mypy** | Type hints | Strict mode |
| **pytest** | Unit tests | (Ready for tests) |

---

## Performance & Device

### Check MPS Support

```python
from src.device import is_mps_available, get_device_info

# Simple check
if is_mps_available():
    print("✓ MPS available")
else:
    print("✗ MPS unavailable, falling back to CPU")

# Detailed info
info = get_device_info()
print(info)
```

### Data Types

| Type | Memory | Speed | Stability | Recommendation |
|------|--------|-------|-----------|-----------------|
| FP32 | 4 bytes | ⭐ | ⭐⭐⭐ | **Use this** |
| FP16 | 2 bytes | ⭐⭐ | ⭐⭐ | Experimental |
| BF16 | 2 bytes | ⭐⭐ | ⭐⭐ | Limited MPS support |

**Recommendation for M1/M2/M3:** Stick with FP32 unless benchmarking shows benefits.

### Benchmarking

```bash
# Compare MPS vs CPU
python scripts/benchmark.py

# Custom benchmark
from src.benchmark import Benchmark
results = Benchmark.compare_devices(
    model_fn=lambda: create_model(...),
    dataloader=train_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer_fn=lambda m: torch.optim.AdamW(m.parameters()),
    devices=["mps", "cpu"],
    dtypes=["fp32"],
    num_steps=100,
)
Benchmark.print_comparison(results)
```

---

## Understanding Runs

### Run Directory Structure

```
runs/20260129_225730_my-run/
├── config_resolved.yaml    # Final configuration used
├── system.json            # Environment info
│   └── {
│       "python_version": "3.13.3",
│       "torch_version": "2.10.0",
│       "git_commit": "3ac34cc...",
│       "device": {"type": "mps", "name": "arm"}
│     }
├── metrics.jsonl          # Metrics per epoch (newline-delimited JSON)
│   └── {"epoch": 0, "train_loss": 0.742, "val_loss": 0.675, ...}
├── train.log              # Full training logs
└── checkpoints/
    ├── checkpoint_epoch_000.pt   # Every epoch
    ├── checkpoint_epoch_001.pt
    ├── checkpoint_epoch_002.pt
    └── best.pt                   # Best by validation loss
```

### Analyzing Metrics

```python
import json

with open("runs/20260129_225730_my-run/metrics.jsonl") as f:
    metrics = [json.loads(line) for line in f]

# Each line is a dictionary
# {"epoch": 0, "train_loss": 0.742, "val_loss": 0.675, "train_time_seconds": 3.23, ...}

for epoch_metrics in metrics:
    print(f"Epoch {epoch_metrics['epoch']}: "
          f"train_loss={epoch_metrics['train_loss']:.4f}, "
          f"val_loss={epoch_metrics['val_loss']:.4f}")
```

---

## Common Issues & Solutions

### Out of Memory (OOM)

**Problem:** `RuntimeError: out of memory`

**Solutions (in order):**
1. Reduce `batch_size` in config
2. Reduce `model.hidden_size`
3. Enable mixed precision (fp16)
4. Use CPU instead of MPS
5. Reduce `dataset.size`

```bash
# Try smaller batch size
python -m src.train --config configs/train.yaml --batch-size 16
```

### MPS Instability

**Problem:** Crashes or numerical errors on MPS

**Solution:** Fall back to CPU

```bash
# Use CPU instead
python -m src.train --config configs/train.yaml --device cpu
```

### Slow Training

**Problem:** Training is slower than expected

**Solution:** Run benchmarks and check device usage

```bash
# Benchmark to understand hardware
python scripts/benchmark.py

# Check if MPS is being used
from src.device import is_mps_available, get_device
device = get_device("mps", fallback_on_error=True)
print(f"Using device: {device}")
```

---

## Advanced Usage

### Custom Models

Create new model in `src/model.py`:

```python
class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Define layers
    
    def forward(self, x):
        # Forward pass
        return output

# Register in model factory
def create_model(...):
    if model_name == "mymodel":
        return MyModel(...)
```

### Custom Training Loop

Modify `src/train.py`:

```python
# Use the exported utilities
from src import (
    train_epoch, evaluate, RunManager, CheckpointManager,
    get_device, get_system_info
)

# Build your own training pipeline
```

### Custom Benchmarks

```python
from src.benchmark import Benchmark
from src.device import get_device

benchmark = Benchmark(
    model=model,
    dataloader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device="mps",
    dtype="fp32",
    warmup_steps=10,
)

results = benchmark.run(num_steps=100, verbose=True)
print(f"Throughput: {results['throughput_samples_per_sec']:.2f} samples/s")
```

---

## MPS Optimization Guide

See [docs/PHASE_E_MPS_GUIDE.md](docs/PHASE_E_MPS_GUIDE.md) for:
- Detailed MPS capabilities and limitations
- Data type tradeoffs (FP32 vs FP16 vs BF16)
- Common OOM patterns and solutions
- Device fallback strategies
- Production-ready training loops

---

## Contributing

### Development Setup

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run quality checks before committing
make all

# Format code
make format
```

### Pull Request Process

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and ensure `make all` passes
3. Commit with clear messages: `git commit -m "Add: description"`
4. Push and create PR

---

## Performance Baseline

**Tested on:** MacBook Pro M1, 16GB RAM

| Config | Device | Throughput | Speedup |
|--------|--------|-----------|---------|
| FP32, batch=32 | CPU | 84,667 samples/s | 13.6x |
| FP32, batch=32 | MPS | 6,207 samples/s | 1.0x |

*Note: Higher samples/s on CPU due to small model and batch size. MPS excels with larger models and batches.*

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)
- [PyTorch Best Practices](https://pytorch.org/tutorials/)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Citation

If you use this template for your project:

```bibtex
@software{pytorch_llm_phase0,
  title={PyTorch LLM Phase 0: Production-Ready Training Pipeline},
  author={{AUTHOR_NAME}},
  year={{YEAR}},
  url={https://github.com/ketcx/pytorch-llm-phase0}
}
```

---

## Support

- 📖 [Full Documentation](docs/)
- 🐛 [Issues](https://github.com/ketcx/pytorch-llm-phase0/issues)
- 💬 [Discussions](https://github.com/ketcx/pytorch-llm-phase0/discussions)

---

<div align="center">

**Made with ❤️ for M1 Mac users learning production ML**

</div>
