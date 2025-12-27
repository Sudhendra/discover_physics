
## 1. Executive Summary

**Objective:** To construct a "Robot Scientist"—an autonomous agent capable of discovering physical laws (specifically the Inverse Square Law of Light, $I \propto 1/d^2$) in a simulated environment without prior programming of those laws.

**Core Philosophy:** Shift autonomous exploration from **Telemetry-First** (collecting data) to **Knowledge-First** (collecting equations). The agent succeeds only when it transmits a compressed mathematical model that accurately predicts the environment, rather than sending raw sensor logs.

## 2. System Architecture

The system is divided into three distinct layers:

1. **The World (Digital Twin):** A high-fidelity physics simulation (MuJoCo + PufferLib) containing the "Hidden Truth."
    
2. **The Analyst (Math Engine):** The symbolic regression stack (PySR) that turns data into candidate equations.
    
3. **The Commander (LLM):** The cognitive layer that interprets the equations, judges scientific validity, and issues high-level stop/continue commands.
    

## 3. Module 1: The Environment (Digital Twin)

**Technology Stack:** `MuJoCo` (Physics), `PufferLib` (Vectorization/Wrapper), `NumPy`.

The environment is designed as a "Black Box" for the agent. It strictly separates the _Physics Engine_ (Truth) from the _Observation Space_ (Noisy Sensors).

**Why MuJoCo:** Free, open-source, excellent macOS ARM support, faster and more accurate than PyBullet, maintained by DeepMind.

### 3.1 Component: The Truth Engine

This logic represents the laws of nature. It is **inaccessible** to the agent's cognitive stack.

```python
import numpy as np

class LightPhysics:
    """
    The 'Hidden Truth' of the universe. 
    The agent must derive the logic inside 'get_true_lux' via experimentation.
    
    This is physics-engine independent - just pure mathematics.
    """
    def __init__(self, source_pos=(0, 0), source_intensity=1000.0):
        self.source_pos = np.array(source_pos, dtype=np.float32)
        self.source_intensity = source_intensity

    def get_true_lux(self, rover_pos):
        # The Inverse Square Law: I = S / (4 * pi * r^2)
        dist_sq = np.sum((rover_pos - self.source_pos)**2)
        dist_sq = max(dist_sq, 0.01)  # Singularity protection
        return self.source_intensity / (4 * np.pi * dist_sq)
```

### 3.2 Component: The MuJoCo Environment

This class defines the agent's physical embodiment and interface. It handles the robot's kinematics in MuJoCo and constructs the noisy observation vector.

**Specifications:**

- **Frequency:** Physics @ 100Hz (MuJoCo default), Control @ 10Hz (approx).
    
- **Arena:** 10m x 10m plane.
    
- **Sensors:** Pose ($x, y$ $\pm$ 1cm error), Lux ($I$ $\pm$ 5% error).
    

```python
import mujoco
import numpy as np
import gymnasium
from gymnasium import spaces

class LuxEnvironment(gymnasium.Env):
    """Gymnasium environment for Lux Scientia using MuJoCo physics."""
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # 1. MuJoCo Model Creation (programmatic - no XML needed for simple rover)
        self.model = self._create_model()
        self.data = mujoco.MjData(self.model)
        
        # 2. Rendering setup
        self.render_mode = render_mode
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # 3. Internal Truth Instantiation
        self.truth = LightPhysics(source_pos=(0, 0))
        
        # 4. Simulation Constraints
        self.tick = 0
        self.max_ticks = 500
        
        # 5. Gymnasium spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, 5000], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
    
    def _create_model(self):
        """Create a simple MuJoCo model programmatically."""
        # Simple 2D rover: box on a plane
        xml_string = """
        <mujoco model="lux_rover">
            <option gravity="0 0 -9.81" timestep="0.01"/>
            
            <worldbody>
                <light pos="0 0 10" dir="0 0 -1"/>
                <geom name="floor" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
                
                <body name="rover" pos="5 5 0.1">
                    <freejoint/>
                    <geom name="rover_geom" type="box" size="0.1 0.1 0.05" 
                          mass="1.0" rgba="0.8 0.2 0.2 1"/>
                </body>
            </worldbody>
            
            <actuator>
                <motor name="force_x" joint="rover" gear="1 0 0 0 0 0"/>
                <motor name="force_y" joint="rover" gear="0 1 0 0 0 0"/>
            </actuator>
        </mujoco>
        """
        return mujoco.MjModel.from_xml_string(xml_string)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Stochastic reset to prevent overfitting
        if seed is not None:
            np.random.seed(seed)
        
        start_x = np.random.uniform(1, 9)
        start_y = np.random.uniform(1, 9)
        
        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = start_x  # x position
        self.data.qpos[1] = start_y  # y position
        self.data.qpos[2] = 0.1      # z position (slightly above ground)
        
        self.tick = 0
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.tick += 1
        
        # Apply forces (scale from [-1,1] to [-10,10] Newtons)
        self.data.ctrl[0] = action[0] * 10.0  # force_x
        self.data.ctrl[1] = action[1] * 10.0  # force_y
        
        # Step physics simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_obs()
        
        # Reward (placeholder for now)
        reward = 0.0
        
        # Episode termination
        terminated = self.tick >= self.max_ticks
        truncated = False
        
        # Render if needed
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()
        
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        """Get noisy sensor observations."""
        # 1. Ground truth from MuJoCo
        raw_x = float(self.data.qpos[0])
        raw_y = float(self.data.qpos[1])
        
        # 2. Sensor noise injection
        obs_x = raw_x + np.random.normal(0, 0.01)
        obs_y = raw_y + np.random.normal(0, 0.01)
        
        # 3. Calculate lux with noise
        true_lux = self.truth.get_true_lux(np.array([raw_x, raw_y]))
        obs_lux = true_lux * (1 + np.random.normal(0, 0.05))
        
        return np.array([obs_x, obs_y, obs_lux], dtype=np.float32)
    
    def close(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
```

### 3.3 PufferLib Vectorization

To allow for rapid "mental simulation" or parallel training, we bind the environment using PufferLib's emulation layer.

```python
import pufferlib
import pufferlib.emulation
import pufferlib.vector

def make_env():
    """Environment creator for PufferLib vectorization."""
    env = LuxEnvironment(render_mode=None)
    return pufferlib.emulation.GymnasiumPufferEnv(env)

# Vectorized environments for parallel execution
vecenv = pufferlib.vector.make(
    env_creator=make_env,
    num_envs=8,
    backend='multiprocessing',
    envs_per_worker=2
)
```

## 4. Module 2: The Scientist Engine (Onboard Logic)

This module runs "on top" of the environment loop. It is the brain of the rover.

### 4.1 Perception Buffer

A sliding window data structure that stores exploration history.

- **Format:** `List[Tuple[float, float, float]]` -> `[(d1, I1), (d2, I2), ...]`
    
- **Preprocessing:** Calculates Euclidean distance from origin (once origin hypothesis is formed) to simplify regression inputs.
    

### 4.2 Symbolic Discovery Engine

Library: PySR (Python Symbolic Regression).

Role: The "Theorist".

- **Configuration:**
    
    - **Binary Operators:** `+, -, *, /`
        
    - **Unary Operators:** `square, inverse` (key for finding $1/d^2$)
        
    - **Complexity Penalty:** High. We want the simplest elegant equation, not an overfitting polynomial.
        
    - **Loss Function:** MSE (Mean Squared Error).
        

### 4.3 Curiosity Controller (The "Strategist")

**Role:** Determines the `action` passed to `env.step()`.

**Algorithm: Surprise-Based Active Learning**

1. **Predict:** Before moving, use the current best PySR Equation to predict Lux at current location.
    
2. **Measure:** Read actual Lux from `obs`.
    
3. **Calculate Surprise:** $\Delta = |Pred - Actual|$.
    
4. **Decision:**
    
    - If $\Delta < Threshold$: Continue Random Walk (Exploration).
        
    - If $\Delta > Threshold$: Trigger **Science Mode**. Stop and sample 5 points in a 10cm radius to verify if the anomaly is noise or a new phenomenon. Update PySR model immediately.
        

## 5. Module 3: The Cognitive Commander (LLM Integration) - [COMPLETED]

**Role:** The LLM does **not** control the rover directly (too slow/costly). Instead, it acts as the **Principal Investigator**. It reviews the output of the Symbolic Regression engine to determine if a "Scientific Discovery" has occurred.

**Stack Choice:** `LiteLLM` (for uniform API calls) + `OpenAI GPT-4o` (as per implementation details).

### 5.1 Technical Integration

- **Trigger Condition:** The LLM is invoked only when `PySR` generates a new "Best Fit" equation with a Complexity Score < 10 and $R^2 > 0.90$.
    
- **Input Context:**
    
    ```
    {
      "current_equation": "y = 800 * (x^-2)",
      "mse_loss": 0.002,
      "variables_identified": ["distance", "intensity"],
      "sample_size": 150
    }
    ```
    
- **System Prompt:** (Defined in `src/commander/protocols.py`)

### 5.2 Implementation (LiteLLM + OpenAI)

Implemented in `src/commander/uplink.py`. It uses `litellm` to wrap the API calls.

## 6. Project Repository Structure

To ensure a clean separation of concerns (Simulation vs. Cognition), the project should follow this directory structure. This organization prevents "Ground Truth" leakage into the Agent's logic.

```
lux-scientia/
├── assets/                     # Optional: MuJoCo XML models (if needed)
│   └── rover.xml               # Custom rover model (optional)
├── config/                     # Configuration Management
│   ├── simulation.yaml         # Physics constants (hidden from agent in theory)
│   └── agent_params.yaml       # PySR hyperparameters, surprise thresholds
├── src/
│   ├── environment/            # Module 1: The Digital Twin
│   │   ├── __init__.py
│   │   ├── physics_engine.py   # Contains 'LightPhysics' (The Truth)
│   │   └── puffer_wrapper.py   # Contains 'LuxEnvironment' & MuJoCo/PufferLib bindings
│   ├── agent/                  # Module 2: The Scientist
│   │   ├── __init__.py
│   │   ├── perception.py       # Rolling buffer & data preprocessing
│   │   ├── theorist.py         # PySR integration & Equation management
│   │   └── navigator.py        # Curiosity Controller & Surprise logic
│   └── commander/              # Module 3: The LLM Interface
│       ├── __init__.py
│       ├── uplink.py           # LiteLLM client wrapper
│       └── protocols.py        # Prompt templates (Discovery vs. Continue)
├── main.py                     # Entry point (runs the active learning loop)
├── pyproject.toml              # UV/Python project config
├── requirements.txt            # mujoco, pufferlib, pysr, litellm, google-generativeai
└── .env                        # GEMINI_API_KEY=...
```

## 7. Execution Phases

### Phase 1: The Digital Bedrock

- **Task:** Implement `LuxEnvironment` and visualize the rover moving in MuJoCo.
    
- **Validation:** Run random walk policy and log the noisy Lux data to a CSV.
    
- **Sanity Check:** Feed this CSV into `PySR` offline to ensure the Inverse Square Law _can_ be recovered from the noisy data.
    
**MuJoCo Advantages:** Faster simulation, better ARM Mac support, cleaner API, industry-standard for robotics RL.
    

### Phase 2: The Loop

- **Task:** Close the loop. Connect the PufferLib `step` output to the `Perception Buffer`.
    
- **Automation:** Run the simulation with a simple "Random Walk" policy.
    
- **Output:** The agent should periodically print "Current Hypothesis: $y = ...$" to the console.
    

### Phase 3: Active Inference

- **Task:** Implement the Curiosity Controller with Active Learning.
- **Next Step:** Implement the LLM Commander to validate hypotheses.
- **Future Optimization:** Replace heuristic Navigator with PPO (Reinforcement Learning) policy trained via PufferLib to maximize the composite reward signal (Safety + Coverage + Curiosity).

- **Objective:** Demonstrate that the agent discovers the law **faster** (fewer steps) using Surprise-based navigation than it did with Random Walk.
    

## 8. Success Criteria

The project is considered a success if:

1. **Discovery:** The Agent outputs a string functionally equivalent to $I = C / d^2$.
    
2. **Autonomy:** The discovery happens without human intervention or "ground truth" leakage.
    
3. **Generalization:** The derived equation accurately predicts Lux values in visited corners of the map ($R^2 > 0.98$).