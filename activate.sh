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
python -c "import numpy, gymnasium, pufferlib, pytest, mujoco; print(f'  - numpy {numpy.__version__}'); print(f'  - gymnasium {gymnasium.__version__}'); print(f'  - pufferlib {pufferlib.__version__}'); print(f'  - pytest {pytest.__version__}'); print(f'  - mujoco {mujoco.__version__}')"
echo ""
echo "✓ All dependencies installed successfully!"
echo "  Physics engine: MuJoCo ${mujoco.__version__} (ARM-native)"
echo ""
echo "Quick start:"
echo "  pytest tests/ -v          # Run all tests"
echo "  python main.py --phase=1  # Run Phase 1"
echo ""
echo "To deactivate: deactivate"
