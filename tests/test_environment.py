"""
Tests for the physics engine and environment.

These tests verify that the ground truth physics and environment
implementation are correct before running experiments.
"""

# Standard library imports
import numpy as np
import pytest

# Local imports
from src.environment.physics_engine import LightPhysics


class TestLightPhysics:
    """Test suite for the LightPhysics truth engine."""
    
    def test_inverse_square_law(self):
        """Verify physics engine implements correct inverse square law."""
        physics = LightPhysics(source_pos=(0, 0), source_intensity=1000.0)
        
        # Test at known distances from origin
        lux_1m = physics.get_true_lux(np.array([1.0, 0.0]))
        lux_2m = physics.get_true_lux(np.array([2.0, 0.0]))
        lux_3m = physics.get_true_lux(np.array([3.0, 0.0]))
        
        # Should follow 1/d^2 relationship
        # At 2m, intensity should be 1/4 of intensity at 1m
        assert np.isclose(lux_1m / lux_2m, 4.0, rtol=0.01), \
            f"Expected 4x ratio, got {lux_1m / lux_2m}"
        
        # At 3m, intensity should be 1/9 of intensity at 1m
        assert np.isclose(lux_1m / lux_3m, 9.0, rtol=0.01), \
            f"Expected 9x ratio, got {lux_1m / lux_3m}"
    
    def test_singularity_protection(self):
        """Verify singularity protection prevents division by zero."""
        physics = LightPhysics(source_pos=(0, 0), source_intensity=1000.0)
        
        # Test at origin (should not crash or return inf/nan)
        lux_origin = physics.get_true_lux(np.array([0.0, 0.0]))
        
        assert np.isfinite(lux_origin), "Lux at origin should be finite"
        assert lux_origin > 0, "Lux at origin should be positive"
    
    def test_radial_symmetry(self):
        """Verify light intensity is radially symmetric."""
        physics = LightPhysics(source_pos=(0, 0), source_intensity=1000.0)
        
        # Test points at same distance but different directions
        lux_north = physics.get_true_lux(np.array([0.0, 2.0]))
        lux_east = physics.get_true_lux(np.array([2.0, 0.0]))
        lux_northeast = physics.get_true_lux(np.array([np.sqrt(2), np.sqrt(2)]))
        
        # All should have same intensity (within numerical precision)
        assert np.isclose(lux_north, lux_east, rtol=1e-5), \
            "Radial symmetry violated"
        assert np.isclose(lux_north, lux_northeast, rtol=1e-5), \
            "Radial symmetry violated"
    
    def test_different_source_positions(self):
        """Verify physics works with non-origin source positions."""
        physics = LightPhysics(source_pos=(5, 5), source_intensity=1000.0)
        
        # Test at known distances from (5, 5)
        lux_at_source = physics.get_true_lux(np.array([5.0, 5.0]))
        lux_1m_away = physics.get_true_lux(np.array([6.0, 5.0]))
        
        # Closer to source should have higher intensity
        assert lux_at_source > lux_1m_away, \
            "Intensity should decrease with distance"
    
    def test_intensity_scaling(self):
        """Verify light intensity scales linearly with source intensity."""
        physics_weak = LightPhysics(source_pos=(0, 0), source_intensity=100.0)
        physics_strong = LightPhysics(source_pos=(0, 0), source_intensity=1000.0)
        
        pos = np.array([2.0, 0.0])
        lux_weak = physics_weak.get_true_lux(pos)
        lux_strong = physics_strong.get_true_lux(pos)
        
        # 10x intensity source should give 10x lux
        assert np.isclose(lux_strong / lux_weak, 10.0, rtol=0.01), \
            "Intensity should scale linearly with source strength"


# Integration tests for the full environment would go here
# But require PyBullet to be installed, so we'll add them after dependencies
