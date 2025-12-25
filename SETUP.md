# Lux Scientia - Setup Guide (UV Package Manager)

This guide walks you through setting up the Lux Scientia project using the **uv** package manager.

## Environment Setup

### ✅ Completed Setup

The following has been configured for you:

1. **UV Package Manager**: Installed at `~/.local/bin/uv`
2. **Virtual Environment**: Created at `.venv/`
3. **Dependencies Installed**:
   - ✅ numpy 1.26.4
   - ✅ gymnasium 0.29.1
   - ✅ pufferlib 3.0.0
   - ✅ pytest 9.0.2
   - ✅ pytest-cov 7.0.0
   - ✅ pyyaml 6.0.3
   - ✅ python-dotenv 1.2.1
   - ⚠️  pybullet (build failed - see below)

## Quick Start

### Activate Environment

```bash
# Option 1: Use activation script
source activate.sh

# Option 2: Manual activation
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PATH="$HOME/.local/bin:$PATH"
```

### Run Tests

```bash
# Test physics engine (works without PyBullet)
pytest tests/test_environment.py::TestLightPhysics -v

# All physics tests should PASS ✓
```

### Verify Installation

```bash
python -c "import numpy, gymnasium, pufferlib, pytest; print('✓ All core dependencies loaded')"
```

## PyBullet Installation (⚠️ Known Issue)

PyBullet failed to build from source on macOS ARM. This is a known issue with PyBullet on Apple Silicon.

### Option 1: Install via Conda (Recommended)

```bash
# Install conda/mamba if not already installed
# Then:
conda install -c conda-forge pybullet

# Or with mamba (faster):
mamba install -c conda-forge pybullet
```

### Option 2: Use Docker

```bash
# Use x86 Docker image with Rosetta 2
docker run -it --platform linux/amd64 python:3.10
# Then install pybullet inside container
```

### Option 3: Wait for ARM Wheel

Check PyPI for ARM-compatible wheels:
```bash
pip index versions pybullet
```

### Option 4: Work Without PyBullet (Testing Only)

For development and testing of the physics engine:
```bash
# Physics tests work without PyBullet
pytest tests/test_environment.py::TestLightPhysics -v
```

## Project Structure

```
lux-scientia/
├── .venv/                 # UV virtual environment
├── src/
│   ├── environment/       # Physics engine + environment
│   ├── agent/            # Perception, reasoning, navigation
│   └── commander/        # LLM interface
├── tests/                # Test suite
├── activate.sh           # Environment activation script
├── pyproject.toml        # UV/Python project config
└── README.md            # Project documentation
```

## UV Commands Reference

### Package Management

```bash
# Add a new dependency
uv pip install package-name

# Add development dependency
uv pip install --dev package-name

# Update all packages
uv pip install --upgrade -e ".[dev]"

# List installed packages
uv pip list

# Show package info
uv pip show package-name
```

### Environment Management

```bash
# Create new venv
uv venv

# Activate venv
source .venv/bin/activate

# Deactivate
deactivate

# Remove venv
rm -rf .venv
```

## Testing

### Run All Tests

```bash
source activate.sh
pytest
```

### Run Specific Tests

```bash
# Physics engine tests
pytest tests/test_environment.py::TestLightPhysics -v

# Specific test
pytest tests/test_environment.py::TestLightPhysics::test_inverse_square_law -v

# With coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Development Workflow

### 1. Activate Environment

```bash
source activate.sh
```

### 2. Make Changes

Edit files in `src/`, `tests/`, etc.

### 3. Run Tests

```bash
pytest -v
```

### 4. Check Code Quality

```bash
# Type checking (if mypy installed)
mypy src/

# Linting (if installed)
flake8 src/ tests/
pylint src/
```

## Troubleshooting

### Import Errors

If you get "ModuleNotFoundError":

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="$PWD:$PYTHONPATH"

# Or always use activate.sh
source activate.sh
```

### UV Not Found

```bash
# Add UV to PATH
export PATH="$HOME/.local/bin:$PATH"

# Or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Virtual Environment Issues

```bash
# Recreate venv
rm -rf .venv
uv venv
uv pip install -e ".[dev]"
```

## Next Steps

1. **Install PyBullet** (via conda recommended)
2. **Run Full Test Suite**: `pytest -v`
3. **Try Phase 1**: `python main.py --phase=1`
4. **Read Documentation**: See `PLAN.md` and `AGENTS.md`

## Support

- **UV Documentation**: https://github.com/astral-sh/uv
- **Project Issues**: Check `PLAN.md` for architecture details
- **PyBullet Issues**: https://github.com/bulletphysics/bullet3/issues

---

**Status**: ✅ Environment configured, ⚠️  PyBullet pending
**Last Updated**: December 2024
