"""
Navigator - Curiosity Controller

This module determines the rover's actions based on exploration strategy.
For Phase 2: Simple random walk
For Phase 3: Surprise-based active learning (curiosity-driven)
"""

# Standard library imports
from typing import Optional, Tuple

# Third-party imports
import numpy as np


class Navigator:
    """
    Controls rover navigation strategy.
    
    Phase 2: Random walk exploration
    Phase 3: Surprise-based active learning (curiosity-driven)
    
    Attributes:
        mode: Current navigation mode ('random' or 'curious')
        surprise_threshold: Threshold for entering "Science Mode"
        science_mode: Whether currently in detailed sampling mode
    """
    
    def __init__(self, mode: str = "random", surprise_threshold: float = 0.15):
        """Initialize navigator.
        
        Args:
            mode: Navigation mode - 'random' for random walk, 'curious' for surprise-based
            surprise_threshold: Threshold for triggering Science Mode (Phase 3)
        """
        self.mode = mode
        self.surprise_threshold = surprise_threshold
        self.science_mode = False
        self.science_samples_remaining = 0
    
    def get_action(self, observation: Optional[np.ndarray] = None, 
                   predicted_lux: Optional[float] = None,
                   unexplored_direction: Optional[np.ndarray] = None) -> np.ndarray:
        """Get next action based on navigation strategy.
        
        Args:
            observation: Current observation [x, y, lux]
            predicted_lux: Predicted lux from theorist model (for curiosity mode)
            unexplored_direction: Direction toward unexplored regions (for active learning)
            
        Returns:
            Action array [force_x, force_y] in range [-1, 1]
        """
        if self.mode == "random":
            return self._random_walk()
        elif self.mode == "curious":
            return self._curiosity_driven(observation, predicted_lux)
        elif self.mode == "active":
            return self._active_learning(observation, predicted_lux, unexplored_direction)
        else:
            return self._random_walk()
    
    def _random_walk(self) -> np.ndarray:
        """Generate random walk action.
        
        Returns:
            Random force vector in range [-1, 1]
        """
        return np.random.uniform(-1, 1, size=2).astype(np.float32)
    
    def _curiosity_driven(self, observation: Optional[np.ndarray], 
                         predicted_lux: Optional[float]) -> np.ndarray:
        """Surprise-based exploration (Phase 3).
        
        Algorithm from PLAN.md 4.3:
        1. Calculate surprise = |predicted - actual|
        2. If surprise > threshold: Enter Science Mode
        3. In Science Mode: Sample 5 points in 10cm radius
        4. Otherwise: Continue random walk
        
        Args:
            observation: Current [x, y, lux]
            predicted_lux: Model's prediction for current position
            
        Returns:
            Action based on surprise level
        """
        # If no prediction available, fall back to random walk
        if observation is None or predicted_lux is None:
            return self._random_walk()
        
        actual_lux = float(observation[2])
        surprise = abs(predicted_lux - actual_lux)
        
        # Check if we should enter Science Mode
        if surprise > self.surprise_threshold and not self.science_mode:
            print(f"⚠  High surprise detected: {surprise:.3f} > {self.surprise_threshold}")
            print(f"   Predicted: {predicted_lux:.2f}, Actual: {actual_lux:.2f}")
            print(f"   Entering Science Mode: sampling 5 points in 10cm radius")
            self.science_mode = True
            self.science_samples_remaining = 5
        
        # If in Science Mode, sample nearby
        if self.science_mode:
            action = self._sample_nearby()
            self.science_samples_remaining -= 1
            
            if self.science_samples_remaining <= 0:
                print(f"✓  Science Mode complete. Resuming exploration.")
                self.science_mode = False
            
            return action
        
        # Otherwise, continue exploration
        return self._random_walk()
    
    def _sample_nearby(self, radius: float = 0.10) -> np.ndarray:
        """Generate small random movement for detailed sampling.
        
        Args:
            radius: Maximum movement radius in meters
            
        Returns:
            Small random action for nearby sampling
        """
        # Small random movement (scaled down for 10cm radius sampling)
        action = np.random.uniform(-0.2, 0.2, size=2).astype(np.float32)
        return action
    
    def calculate_surprise(self, predicted: float, actual: float) -> float:
        """Calculate prediction error (surprise metric).
        
        Uses Relative Error instead of Absolute Error to account for 
        multiplicative sensor noise (heteroscedasticity).
        
        Args:
            predicted: Model's predicted lux value
            actual: Observed lux from sensor
            
        Returns:
            Relative error: |pred - actual| / (pred + epsilon)
        """
        # Add small epsilon to prevent division by zero
        epsilon = 1e-6
        absolute_diff = abs(predicted - actual)
        # Calculate relative error (scale-invariant)
        relative_error = absolute_diff / (abs(predicted) + epsilon)
        
        return relative_error
    
    def should_enter_science_mode(self, predicted: float, actual: float) -> bool:
        """Determine if anomaly warrants detailed investigation.
        
        Args:
            predicted: Predicted lux value
            actual: Actual lux value
            
        Returns:
            True if surprise exceeds threshold
        """
        surprise = self.calculate_surprise(predicted, actual)
        return surprise > self.surprise_threshold
    
    def reset_science_mode(self):
        """Reset science mode state."""
        self.science_mode = False
        self.science_samples_remaining = 0
    
    def set_mode(self, mode: str):
        """Change navigation mode.
        
        Args:
            mode: 'random', 'curious', or 'active'
        """
        self.mode = mode
        self.reset_science_mode()
    
    def _active_learning(self, observation: Optional[np.ndarray],
                        predicted_lux: Optional[float],
                        unexplored_direction: Optional[np.ndarray]) -> np.ndarray:
        """Active learning exploration - uncertainty and coverage driven.
        
        Combines:
        1. Surprise-based sampling (like curiosity)
        2. Coverage-based exploration (toward unexplored regions)
        
        Args:
            observation: Current [x, y, lux]
            predicted_lux: Model prediction
            unexplored_direction: Direction toward unexplored cells
            
        Returns:
            Action balancing surprise and coverage
        """
        # If in Science Mode (high surprise), sample nearby
        if self.science_mode:
            action = self._sample_nearby()
            self.science_samples_remaining -= 1
            
            if self.science_samples_remaining <= 0:
                self.science_mode = False
            
            return action
        
        # Check for high surprise
        if observation is not None and predicted_lux is not None:
            actual_lux = float(observation[2])
            surprise = abs(predicted_lux - actual_lux)
            
            if surprise > self.surprise_threshold:
                print(f"  ⚠  Surprise: {surprise:.3f} > {self.surprise_threshold} - Science Mode")
                self.science_mode = True
                self.science_samples_remaining = 5
                return self._sample_nearby()
        
        # Otherwise, bias toward unexplored regions
        if unexplored_direction is not None:
            # 70% toward unexplored, 30% random for exploration-exploitation balance
            bias_strength = 0.7
            random_component = np.random.uniform(-1, 1, size=2).astype(np.float32)
            biased_action = bias_strength * unexplored_direction + (1 - bias_strength) * random_component
            
            # Normalize and clip
            norm = np.linalg.norm(biased_action)
            if norm > 0:
                biased_action = biased_action / norm
            
            return np.clip(biased_action, -1.0, 1.0).astype(np.float32)
        
        # Fallback to random walk
        return self._random_walk()
    
    def calculate_intrinsic_reward(self, observation: Optional[np.ndarray], 
                                 predicted_lux: Optional[float],
                                 coverage_reward: float) -> float:
        """Calculate total intrinsic reward (Curiosity + Coverage).
        
        Args:
            observation: Current [x, y, lux]
            predicted_lux: Model prediction
            coverage_reward: Reward from PerceptionBuffer [0.0, 1.0]
            
        Returns:
            Total intrinsic reward scalar
        """
        curiosity_reward = 0.0
        
        # Calculate Curiosity Reward (Scaled Surprise)
        if observation is not None and predicted_lux is not None:
            actual_lux = float(observation[2])
            surprise = self.calculate_surprise(predicted_lux, actual_lux)
            
            # Bonus: 0.5 * RelativeSurprise
            curiosity_reward = 0.5 * surprise
            
        # Combine: Coverage (Exploration) + Curiosity (Science)
        # Weighting: 1.0 for Coverage (Primary), 0.5 for Curiosity (Secondary)
        total_reward = coverage_reward + curiosity_reward
        
        return total_reward
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Navigator(mode={self.mode}, science_mode={self.science_mode})"
