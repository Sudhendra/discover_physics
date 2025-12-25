"""
Integration tests for the complete environment with MuJoCo.

These tests verify the full perception-action loop works correctly.
"""

# Standard library imports
import numpy as np
import pytest

# Third-party imports
import pufferlib
import pufferlib.emulation
import pufferlib.vector

# Local imports
from src.environment.puffer_wrapper import LuxEnvironment


class TestLuxEnvironment:
    """Test suite for the MuJoCo-based LuxEnvironment."""
    
    def test_environment_creation(self):
        """Test that environment can be created successfully."""
        env = LuxEnvironment(render_mode=None)
        assert env is not None
        assert env.observation_space.shape == (3,)
        assert env.action_space.shape == (2,)
        env.close()
    
    def test_reset(self):
        """Test environment reset functionality."""
        env = LuxEnvironment(render_mode=None)
        obs, info = env.reset(seed=42)
        
        # Check observation shape and bounds
        assert obs.shape == (3,), f"Expected shape (3,), got {obs.shape}"
        assert 0 <= obs[0] <= 10, f"x position out of bounds: {obs[0]}"
        assert 0 <= obs[1] <= 10, f"y position out of bounds: {obs[1]}"
        assert obs[2] > 0, f"lux should be positive, got {obs[2]}"
        
        env.close()
    
    def test_step(self):
        """Test single environment step."""
        env = LuxEnvironment(render_mode=None)
        obs, _ = env.reset(seed=42)
        
        # Take action
        action = np.array([0.5, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify outputs
        assert obs.shape == (3,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        
        env.close()
    
    def test_multiple_steps(self):
        """Test multiple steps and episode completion."""
        env = LuxEnvironment(render_mode=None, max_ticks=10)
        obs, _ = env.reset(seed=42)
        
        for i in range(15):  # More than max_ticks
            action = np.random.uniform(-1, 1, size=2).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # After max_ticks, should autoreset
            if i >= 10:
                # Environment should have autoreset
                assert obs.shape == (3,)
        
        env.close()
    
    def test_deterministic_reset(self):
        """Test that reset with same seed gives same initial state."""
        env1 = LuxEnvironment(render_mode=None)
        env2 = LuxEnvironment(render_mode=None)
        
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        
        # Note: observations have noise, but positions should be similar
        # (noise is applied after reset, so initial positions should match)
        np.testing.assert_allclose(obs1[:2], obs2[:2], rtol=0.1)
        
        env1.close()
        env2.close()
    
    def test_action_bounds(self):
        """Test that actions outside bounds are handled correctly."""
        env = LuxEnvironment(render_mode=None)
        obs, _ = env.reset(seed=42)
        
        # Test extreme actions (should be clipped)
        extreme_action = np.array([100.0, -100.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(extreme_action)
        
        # Should not crash and should return valid observation
        assert obs.shape == (3,)
        assert np.all(np.isfinite(obs))
        
        env.close()


class TestPufferLibIntegration:
    """Test PufferLib integration with LuxEnvironment."""
    
    def test_single_env_wrapper(self):
        """Test wrapping single environment with PufferLib."""
        def make_env():
            env = LuxEnvironment(render_mode=None)
            return pufferlib.emulation.GymnasiumPufferEnv(env)
        
        env = make_env()
        obs, info = env.reset(seed=42)
        
        # PufferLib adds batch dimension even for single env
        assert obs.shape == (1, 3), f"Expected (1, 3), got {obs.shape}"
        
        action = np.array([[0.5, 0.5]], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (1, 3)
        env.close()
    
    def test_vectorized_serial_backend(self):
        """Test vectorized environment with Serial backend."""
        def make_env():
            env = LuxEnvironment(render_mode=None, max_ticks=100)
            return pufferlib.emulation.GymnasiumPufferEnv(env)
        
        # Create vectorized environment with Serial backend
        vecenv = pufferlib.vector.make(
            make_env,
            backend=pufferlib.vector.Serial,
            num_envs=2,
            seed=42
        )
        
        # Reset
        obs, _ = vecenv.reset(seed=42)
        assert obs.shape == (2, 3), f"Expected (2, 3), got {obs.shape}"
        
        # Step
        actions = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
        obs, rewards, terminateds, truncateds, infos = vecenv.step(actions)
        
        assert obs.shape == (2, 3)
        assert rewards.shape == (2,)
        
        vecenv.close()
    
    def test_vectorized_multiprocessing_backend(self):
        """Test multiple steps with Multiprocessing backend."""
        def make_env():
            env = LuxEnvironment(render_mode=None, max_ticks=50)
            return pufferlib.emulation.GymnasiumPufferEnv(env)
        
        # Create vectorized environment
        vecenv = pufferlib.vector.make(
            make_env,
            backend=pufferlib.vector.Multiprocessing,
            num_envs=4,
            seed=42
        )
        
        # Reset
        obs, _ = vecenv.reset(seed=42)
        assert obs.shape == (4, 3)
        
        # Run a few steps
        for _ in range(5):
            actions = np.random.uniform(-1, 1, size=(4, 2)).astype(np.float32)
            obs, rewards, terminateds, truncateds, infos = vecenv.step(actions)
            assert obs.shape == (4, 3)
            assert rewards.shape == (4,)
        
        vecenv.close()
