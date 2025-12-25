#!/bin/bash
# Lux Scientia - Environment Activation Script
#
# Usage: source activate.sh

# Activate virtual environment
source .venv/bin/activate

# Add project root to PYTHONPATH
export PYTHONPATH="/Users/sudhendrakambhamettu/Desktop/projects/explore_away:$PYTHONPATH"

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

echo "✓ Lux Scientia environment activated"
echo "  Python: $(python --version)"
echo "  Location: $(which python)"
echo ""
echo "Installed packages:"
python -c "import numpy, gymnasium, pufferlib, pytest; print(f'  - numpy {numpy.__version__}'); print(f'  - gymnasium {gymnasium.__version__}'); print(f'  - pufferlib {pufferlib.__version__}'); print(f'  - pytest {pytest.__version__}')"
echo ""
echo "⚠  Note: PyBullet build failed on macOS ARM. Alternatives:"
echo "  1. Install via conda: conda install -c conda-forge pybullet"
echo "  2. Use without physics: Run tests with mock physics"
echo "  3. Wait for ARM-compatible wheel"
echo ""
echo "To deactivate: deactivate"
