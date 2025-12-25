# Lux Scientia - Autonomous Robot Scientist

An autonomous agent that discovers physical laws (specifically the Inverse Square Law of Light) through experimentation in a simulated environment, without prior programming of those laws.

## Philosophy

**Knowledge-First Exploration**: The agent succeeds by discovering compressed mathematical models (equations), not by collecting raw data. Success is measured by transmitting an equation like `I = C/d²`, not sensor logs.

## Project Structure

```
lux-scientia/
├── src/
│   ├── environment/     # Digital Twin (PyBullet simulation)
│   │   ├── physics_engine.py   # LightPhysics (hidden truth)
│   │   └── puffer_wrapper.py   # LuxEnvironment (agent interface)
│   ├── agent/          # Scientist (perception, reasoning, navigation)
│   └── commander/      # LLM Interface (discovery validation)
├── tests/              # Unit and integration tests
├── config/             # Configuration files
├── main.py             # Entry point
└── PLAN.md             # Complete design document
```

## Quick Start

### 1. Activate Environment (UV)

```bash
# Activate virtual environment and set paths
source activate.sh
```

**Note**: This project uses **UV** package manager. A virtual environment has been pre-configured at `.venv/`.

### 2. Verify Installation

All dependencies including MuJoCo should already be installed. Verify:

```bash
source activate.sh
# Should show all packages including mujoco 3.4.0
```

### 3. Run Phase 1 (Environment Validation)

```bash
# Without visualization (faster)
python main.py --phase=1

# With PyBullet GUI (to see the rover)
python main.py --phase=1 --render

# Longer run
python main.py --phase=1 --render --steps=500
```

### 4. Run Tests

```bash
# Activate environment first
source activate.sh

# Run physics engine tests
pytest tests/test_environment.py::TestLightPhysics -v

# Run all tests (includes MuJoCo environment tests)
pytest -v

# Run with coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

**Status**: ✅ All dependencies installed, including MuJoCo 3.4.0 (ARM-native).

## Implementation Phases

### Phase 1: Digital Bedrock ✅

**Status**: Implemented

**Goals**:
- Implement `LuxEnvironment` with PyBullet
- Visualize rover movement
- Collect noisy sensor data
- Validate physics engine

**Run**:
```bash
python main.py --phase=1 --render
```

### Phase 2: The Loop (Coming Soon)

**Goals**:
- Connect PufferLib vectorization
- Implement Perception Buffer
- Run random walk policy
- Periodically output hypothesis equations

**Run**:
```bash
python main.py --phase=2 --save-csv
```

### Phase 3: Active Inference (Coming Soon)

**Goals**:
- Implement PySR symbolic regression
- Add Curiosity Controller (surprise-based navigation)
- Integrate LLM for discovery validation
- Demonstrate faster discovery vs random walk

**Run**:
```bash
python main.py --phase=3
```

## Architecture

### Three-Layer System

1. **The World (Digital Twin)**
   - MuJoCo physics simulation
   - Hidden ground truth: Inverse Square Law
   - 10m × 10m arena

2. **The Analyst (Math Engine)**
   - PySR symbolic regression
   - Discovers equations from noisy data
   - Complexity-penalized optimization

3. **The Commander (LLM)**
   - Google Gemini via LiteLLM
   - Validates discovered equations
   - Issues continue/discovery commands

### Information Barrier

**Critical**: The agent NEVER has direct access to ground truth.

- Physics engine (`LightPhysics`) contains hidden truth
- Agent only receives noisy sensor observations
- PufferLib wrapper reinforces this barrier
- Success = discovering the law without cheating

## Environment Specifications

- **Arena**: 10m × 10m plane
- **Physics Frequency**: 240 Hz
- **Control Frequency**: ~10 Hz
- **Sensors**:
  - Position (x, y): ±1cm noise
  - Light intensity (lux): ±5% noise
- **Actions**: Continuous force vector [-1, 1] → Newtons

## Success Criteria

The project succeeds when:

1. **Discovery**: Agent outputs equation functionally equivalent to `I = C/d²`
2. **Autonomy**: Discovery happens without human intervention or ground truth leakage
3. **Generalization**: Equation accurately predicts lux values (R² > 0.98)

## Development

### Code Style

- **Classes**: PascalCase (`LightPhysics`, `LuxEnvironment`)
- **Functions**: snake_case (`get_true_lux`, `calculate_surprise`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_TICKS`, `SURPRISE_THRESHOLD`)
- **Docstrings**: Google-style with type hints

See `AGENTS.md` for complete style guide.

### Testing

```bash
# Unit tests (physics engine)
pytest tests/test_environment.py -v

# Integration tests (full environment loop)
pytest tests/test_integration.py -v

# Coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## References

- **Design Document**: [PLAN.md](PLAN.md)
- **Agent Guide**: [AGENTS.md](AGENTS.md)
- **PySR**: https://astroautomata.com/PySR/
- **PufferLib**: https://github.com/PufferAI/PufferLib
- **PyBullet**: https://pybullet.org/

## License

MIT License - See LICENSE file for details.

## Contributing

This is a research project. See `PLAN.md` for the complete vision and `AGENTS.md` for development guidelines.
