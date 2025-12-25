# Agent Development Guide - Lux Scientia

This document provides essential information for AI coding agents working on the Lux Scientia project—an autonomous Robot Scientist that discovers physical laws through experimentation in a simulated environment.

## Project Overview

**Language**: Python 3.10+  
**Type**: Scientific simulation with symbolic AI and LLM integration  
**Core Stack**: MuJoCo (physics), PufferLib (vectorization), PySR (symbolic regression), LiteLLM + Google Gemini (reasoning)

**Physics Engine**: MuJoCo - Free, open-source, excellent macOS ARM support, faster and more accurate than PyBullet

## Build & Run Commands

### Setup
```bash
# Activate UV environment
source activate.sh

# Install MuJoCo and dependencies (if not already done)
uv pip install mujoco mujoco-python-viewer

# Set up environment variables
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_api_key_here
```

### Running the Project
```bash
# Activate environment first
source activate.sh

# Main entry point
python main.py

# With visualization (MuJoCo viewer)
python main.py --render

# Specific phases
python main.py --phase=1  # Digital Bedrock validation
python main.py --phase=2  # Loop testing
python main.py --phase=3  # Active inference
```

### Testing
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_environment.py

# Run a specific test function
pytest tests/test_environment.py::test_inverse_square_law

# Run with coverage
pytest --cov=src --cov-report=html

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_lux"

# Test vectorization (serial backend for debugging)
pytest tests/test_vectorization.py -v
```

### Linting & Formatting
```bash
# Format code (if black is configured)
black src/ tests/

# Sort imports (if isort is configured)
isort src/ tests/

# Type checking (if mypy is configured)
mypy src/

# Lint (if flake8/pylint is configured)
flake8 src/ tests/
pylint src/
```

## Code Style Guidelines

### File Organization
Follow the planned structure strictly to prevent "ground truth" leakage (see PLAN.md section 6):
```
src/
├── environment/     # Digital Twin (physics simulation) - Contains TRUTH
│   ├── physics_engine.py   # LightPhysics (hidden truth)
│   └── puffer_wrapper.py   # LuxEnvironment & PufferLib bindings
├── agent/          # Scientist (perception, reasoning, navigation)
│   ├── perception.py       # Rolling buffer & preprocessing
│   ├── theorist.py         # PySR integration
│   └── navigator.py        # Curiosity Controller
└── commander/      # LLM Interface (discovery validation)
    ├── uplink.py           # LiteLLM client wrapper
    └── protocols.py        # Prompt templates
```

### Import Organization
```python
# Standard library imports
import os
from typing import Tuple, List, Optional

# Third-party imports
import numpy as np
import mujoco
import gymnasium
import pufferlib
import pufferlib.emulation
import pufferlib.vector
from litellm import completion

# Local imports
from src.environment.physics_engine import LightPhysics
from src.agent.perception import PerceptionBuffer
```

### Naming Conventions

**Classes**: PascalCase
```python
class LightPhysics:
class LuxEnvironment:
class PerceptionBuffer:
```

**Functions/Methods**: snake_case (descriptive, verb-based)
```python
def get_true_lux(rover_pos):
def calculate_surprise(predicted, actual):
def consult_commander(equation, mse):
```

**Constants**: UPPER_SNAKE_CASE
```python
MAX_TICKS = 500
PHYSICS_FREQ = 240
CONTROL_FREQ = 10
SURPRISE_THRESHOLD = 0.15
```

**Private methods**: Leading underscore
```python
def _get_obs(self):
def _inject_noise(self, value, std):
def _preprocess_data(self, buffer):
```

### Type Annotations
Always use type hints for function signatures:
```python
def get_true_lux(self, rover_pos: np.ndarray) -> float:
    """Calculate true light intensity at rover position."""
    dist_sq = np.sum((rover_pos - self.source_pos)**2)
    dist_sq = max(dist_sq, 0.01)  # Singularity protection
    return self.source_intensity / (4 * np.pi * dist_sq)

def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """Reset environment to initial state."""
    if seed:
        np.random.seed(seed)
    return self._get_obs(), {}
```

### Docstrings
Use Google-style docstrings for all public functions/classes:
```python
def calculate_surprise(predicted: float, actual: float) -> float:
    """Calculate prediction error for active learning.
    
    Args:
        predicted: Model's predicted lux value
        actual: Observed lux from sensor
        
    Returns:
        Absolute difference between predicted and actual values
        
    Note:
        High surprise triggers "Science Mode" for detailed sampling
    """
    return abs(predicted - actual)
```

### Error Handling

**Guard against singularities** (critical for physics):
```python
def get_true_lux(self, rover_pos: np.ndarray) -> float:
    dist_sq = np.sum((rover_pos - self.source_pos)**2)
    dist_sq = max(dist_sq, 0.01)  # Singularity protection
    return self.source_intensity / (4 * np.pi * dist_sq)
```

**Handle LLM failures gracefully**:
```python
try:
    response = completion(model="gemini/gemini-1.5-flash", messages=messages)
    content = response.choices[0].message.content
except Exception as e:
    print(f"Commander Uplink Failed: {e}")
    return False  # Default to continuing mission
```

**Validate environment constraints**:
```python
def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
    # Clamp actions to valid range
    action = np.clip(action, -1.0, 1.0)
    
    # Apply forces to MuJoCo
    self.data.ctrl[0] = action[0] * 10.0
    self.data.ctrl[1] = action[1] * 10.0
    
    # Check for boundary violations
    x, y = self.data.qpos[0], self.data.qpos[1]
    if not (0 <= x <= 10 and 0 <= y <= 10):
        # Handle out-of-bounds
        pass
```

## PufferLib Integration

### Environment Creator Pattern
**Critical**: Always use callable creator functions, never pass environment instances directly (PLAN.md 3.3).

```python
def make_env():
    """Environment creator function - returns a new LuxEnvironment instance."""
    return LuxEnvironment(render=False)

# For vectorized environments
vecenv = pufferlib.vector.make(
    env_creator=make_env,  # Pass the function, not an instance
    num_envs=8,
    backend='multiprocessing',
    envs_per_worker=2
)
```

**Key Requirements:**
- Creator function must return a **NEW** environment instance each time
- Function must be **picklable** for multiprocessing backend
- Keep the function **stateless** and **side-effect free**

### Emulation Wrapper
PufferLib's `GymnasiumPufferEnv` wrapper provides automatic space flattening and batching:

```python
import pufferlib.emulation

def make_env():
    env = LuxEnvironment(render=False)
    # Wrap with PufferLib emulation layer
    return pufferlib.emulation.GymnasiumPufferEnv(env)

# Single environment for testing
env = make_env()
obs, info = env.reset()

# Vectorized environments for parallel execution
vecenv = pufferlib.vector.make(make_env, num_envs=8, backend='multiprocessing')
```

**What the Emulation Layer Does:**
- Flattens hierarchical observation/action spaces into 1D arrays
- Handles batching for vectorized environments
- Maintains information barrier (only sees `step()` outputs, never internal physics)
- Provides `single_observation_space` and `single_action_space` attributes

### Vectorization Strategy
Recommended configuration for Lux Scientia:

```python
# Start with serial backend for debugging
vecenv = pufferlib.vector.make(
    make_env,
    num_envs=2,
    backend='serial'  # No parallelization, easier debugging
)

# Scale up with multiprocessing for parallel execution
vecenv = pufferlib.vector.make(
    make_env,
    num_envs=8,
    backend='multiprocessing',  # Recommended for CPU parallelization
    envs_per_worker=2,  # 4 workers × 2 envs each
    seed=42
)
```

**Backend Options:**
- `serial`: No parallelization, best for debugging and initial testing
- `multiprocessing`: Synchronous parallel execution (recommended default)
- `multiprocessing_async`: Asynchronous execution for maximum throughput

### Autoreset Implementation
PufferLib expects environments to handle internal resets on episode termination:

```python
def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
    self.tick += 1
    
    # Apply forces to MuJoCo actuators
    self.data.ctrl[0] = action[0] * 10.0  # force_x
    self.data.ctrl[1] = action[1] * 10.0  # force_y
    
    # Step MuJoCo physics
    mujoco.mj_step(self.model, self.data)
    
    obs = self._get_obs()
    reward = 0.0
    terminated = self.tick >= self.max_ticks
    truncated = False
    
    # Autoreset: Reset internal state when episode ends
    if terminated or truncated:
        # Reset for next episode (keep same MuJoCo model)
        obs, _ = self.reset()
    
    return obs, reward, terminated, truncated, {}
```

### Testing Vectorization
Always test with serial backend first before scaling to multiprocessing:

```python
def test_vectorized_environment():
    """Test environment with PufferLib vectorization."""
    def make_env():
        env = LuxEnvironment(render=False)
        return pufferlib.emulation.GymnasiumPufferEnv(env)
    
    # Test serial first (easier debugging)
    vecenv = pufferlib.vector.make(make_env, num_envs=2, backend='serial')
    obs, _ = vecenv.reset()
    
    assert obs.shape == (2, 3)  # Batched: [num_envs, obs_dim] for [x, y, lux]
    
    # Test step
    actions = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
    obs, rewards, dones, truncs, infos = vecenv.step(actions)
    
    assert rewards.shape == (2,)
    assert obs.shape == (2, 3)
```

## Architecture Principles

### Information Barrier
**Critical**: The agent must NEVER have direct access to ground truth. Maintain strict separation:
- `environment/physics_engine.py` contains truth (inverse square law)
- `agent/` modules only receive noisy observations
- Never pass `LightPhysics` instance to agent code
- PufferLib emulation wrapper reinforces this barrier (only sees `step()` outputs)

### Knowledge-First Philosophy
The agent succeeds by discovering compressed mathematical models, not by collecting data:
```python
# Good: Agent transmits discovered equation
return {"status": "DISCOVERY", "equation": "I = 800 / d^2", "r_squared": 0.98}

# Bad: Agent transmits raw telemetry
return {"status": "DATA", "samples": [(x1,y1,lux1), (x2,y2,lux2), ...]}
```

### Noise Injection
Always inject realistic sensor noise to prevent trivial solutions:
```python
def _get_obs(self) -> np.ndarray:
    # Get position from MuJoCo state
    raw_x = float(self.data.qpos[0])
    raw_y = float(self.data.qpos[1])
    
    # Sensor noise: ±1cm for position, ±5% for lux
    obs_x = raw_x + np.random.normal(0, 0.01)
    obs_y = raw_y + np.random.normal(0, 0.01)
    
    true_lux = self.truth.get_true_lux(np.array([raw_x, raw_y]))
    obs_lux = true_lux * (1 + np.random.normal(0, 0.05))
    
    return np.array([obs_x, obs_y, obs_lux], dtype=np.float32)
```

## Testing Guidelines

### Unit Tests
Test each module in isolation:
```python
def test_inverse_square_law():
    """Verify physics engine implements correct law."""
    physics = LightPhysics(source_pos=(0, 0), source_intensity=1000.0)
    
    # Test at known distances
    lux_1m = physics.get_true_lux(np.array([1.0, 0.0]))
    lux_2m = physics.get_true_lux(np.array([2.0, 0.0]))
    
    # Should follow 1/d^2 relationship
    assert np.isclose(lux_1m / lux_2m, 4.0, rtol=0.01)
```

### Integration Tests
Verify the full perception-action loop with PufferLib wrapper:
```python
def test_environment_loop():
    """Test complete environment step cycle with PufferLib wrapper."""
    def make_env():
        env = LuxEnvironment(render=False)
        return pufferlib.emulation.GymnasiumPufferEnv(env)
    
    # Test with single environment first
    env = make_env()
    obs, info = env.reset()
    
    assert obs.shape == (3,)  # [x, y, lux]
    
    action = np.array([0.5, 0.5], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    
    assert 0 <= obs[0] <= 10  # x in bounds
    assert 0 <= obs[1] <= 10  # y in bounds
    assert obs[2] > 0         # lux is positive
    
    # Test with vectorized environment
    vecenv = pufferlib.vector.make(make_env, num_envs=2, backend='serial')
    obs, _ = vecenv.reset()
    assert obs.shape == (2, 3)  # Batched observations
```

### Validation Tests
Confirm PySR can discover the law from noisy data:
```python
def test_symbolic_regression_discovers_law():
    """Verify PySR recovers inverse square from synthetic data."""
    # Generate noisy samples
    distances = np.linspace(0.5, 5.0, 100)
    true_lux = 1000 / (4 * np.pi * distances**2)
    noisy_lux = true_lux * (1 + np.random.normal(0, 0.05, size=100))
    
    # Run symbolic regression
    model = fit_pysr_model(distances, noisy_lux)
    
    # Check if discovered equation has form 1/x^2
    assert model.complexity < 10
    assert model.r_squared > 0.90
```

## Common Patterns

### Curiosity-Driven Exploration
```python
def should_enter_science_mode(self, predicted: float, actual: float) -> bool:
    """Determine if anomaly warrants detailed investigation."""
    surprise = abs(predicted - actual)
    return surprise > self.SURPRISE_THRESHOLD
```

### LLM Consultation
Only invoke LLM when symbolic regression finds promising equation:
```python
if model.complexity < 10 and model.r_squared > 0.90:
    discovery_confirmed = consult_commander(model.equation, model.mse)
    if discovery_confirmed:
        return True  # Mission success
```

### Environment Creator Pattern
```python
def make_lux_env():
    """Create and wrap LuxEnvironment for PufferLib vectorization."""
    env = LuxEnvironment(
        render=False,
        max_ticks=500
    )
    return pufferlib.emulation.GymnasiumPufferEnv(env)

# Usage in main loop
if __name__ == '__main__':
    vecenv = pufferlib.vector.make(
        make_lux_env,
        num_envs=8,
        backend='multiprocessing',
        envs_per_worker=2
    )
```

## Configuration Management

Load hyperparameters from YAML, never hardcode:
```python
# config/agent_params.yaml
pysr:
  binary_operators: ["+", "-", "*", "/"]
  unary_operators: ["square", "inv"]
  complexity_penalty: 0.1
  
curiosity:
  surprise_threshold: 0.15
  science_mode_samples: 5
  sample_radius: 0.10
```

## Notes for Agents

1. **Read PLAN.md first** - Contains complete system design and philosophy
2. **Preserve information barriers** - Never give agent direct access to LightPhysics
3. **Test physics offline** - Validate inverse square law discovery with CSV data before full integration
4. **Start simple** - Implement Phase 1 (visualization), then Phase 2 (loop), then Phase 3 (curiosity)
5. **Cost-conscious LLM usage** - Only call Gemini when high-confidence equations are found
6. **Deterministic testing** - Use fixed seeds for reproducible simulation testing
7. **PufferLib integration** - Wrap environment correctly to enable vectorization for parallel exploration

## References

- Main design document: `PLAN.md`
- PySR documentation: https://astroautomata.com/PySR/
- PufferLib docs: https://github.com/PufferAI/PufferLib
- MuJoCo documentation: https://mujoco.readthedocs.io/
- MuJoCo Python bindings: https://github.com/google-deepmind/mujoco
