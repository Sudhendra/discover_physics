"""
Theorist - Symbolic Regression Engine

This module wraps PySR to discover mathematical equations from data.
It's the "brain" that converts observations into testable hypotheses.
"""

# Standard library imports
from typing import Optional, Dict, Any
import warnings

# Third-party imports
import numpy as np

# Suppress PySR warnings for cleaner output
warnings.filterwarnings('ignore')


class Theorist:
    """
    Symbolic regression engine using PySR.
    
    Discovers mathematical relationships between distance and light intensity.
    Maintains the current best hypothesis equation.
    
    Attributes:
        model: PySR model (None until first fit)
        best_equation: Best discovered equation string
        best_r_squared: R² score of best equation
        best_complexity: Complexity score of best equation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the theorist.
        
        Args:
            config: PySR configuration parameters (operators, complexity penalty, etc.)
        """
        self.config = config or self._default_config()
        self.model = None
        self.best_equation: Optional[str] = None
        self.best_r_squared: float = 0.0
        self.best_complexity: int = 999
        self.min_samples: int = 50  # Minimum samples before attempting regression
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default PySR configuration.
        
        Returns:
            Default configuration matching PLAN.md section 4.2
        """
        return {
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["square", "inv"],
            "complexity_of_operators": {"square": 1, "inv": 1},
            "complexity_of_constants": 1,
            "parsimony": 0.01,  # Lower penalty for better fits
            "niterations": 40,  # More iterations for convergence
            "populations": 15,
            "population_size": 33,
            "maxsize": 12,  # Slightly lower max complexity
            "verbosity": 0,  # Silent mode
            "denoise": True,  # Handle noisy data better
            "model_selection": "best",  # Use best model
        }
    
    def fit(self, distances: np.ndarray, intensities: np.ndarray) -> bool:
        """Fit symbolic regression model to data.
        
        Args:
            distances: Array of distances from light source
            intensities: Array of corresponding lux readings
            
        Returns:
            True if a good equation was found, False otherwise
            
        Note:
            Only runs if we have enough samples. Updates best_equation if
            a better model is found.
        """
        # Check minimum sample size
        if len(distances) < self.min_samples:
            return False
        
        # Lazy import PySR (optional dependency)
        try:
            from pysr import PySRRegressor
        except ImportError:
            print("⚠  PySR not installed. Install with: uv pip install pysr")
            print("   Skipping symbolic regression (will use random walk only)")
            return False
        
        try:
            # Create and fit PySR model
            model = PySRRegressor(**self.config)
            model.fit(distances.reshape(-1, 1), intensities)
            
            # Get best equation - use actual R² calculation
            if hasattr(model, 'equations_'):
                # Sort by loss (lower is better)
                best_idx = model.equations_['loss'].idxmin()
                equation_row = model.equations_.iloc[best_idx]
                
                equation_str = str(equation_row['equation'])
                loss = float(equation_row.get('loss', float('inf')))
                complexity = int(equation_row.get('complexity', 999))
                
                # Calculate actual R² from predictions
                try:
                    predictions = model.predict(distances.reshape(-1, 1))
                    ss_res = np.sum((intensities - predictions) ** 2)
                    ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                except Exception:
                    r_squared = 0.0
                
                # Update if this is better than current best
                if r_squared > self.best_r_squared or \
                   (r_squared >= self.best_r_squared and complexity < self.best_complexity):
                    self.model = model
                    self.best_equation = equation_str
                    self.best_r_squared = r_squared
                    self.best_complexity = complexity
                    
                    print(f"\n🔬 New Hypothesis: {equation_str}")
                    print(f"   R² = {r_squared:.4f}, Loss = {loss:.6f}, Complexity = {complexity}")
                    
                    return True
            
        except Exception as e:
            print(f"⚠  PySR fitting failed: {e}")
            return False
        
        return False
    
    def predict(self, distance: float) -> Optional[float]:
        """Predict intensity at given distance using current model.
        
        Args:
            distance: Distance from light source in meters
            
        Returns:
            Predicted intensity, or None if no model exists
        """
        if self.model is None:
            return None
        
        try:
            prediction = self.model.predict(np.array([[distance]]))
            return float(prediction[0])
        except Exception:
            return None
    
    def has_good_hypothesis(self, min_r_squared: float = 0.90, max_complexity: int = 10) -> bool:
        """Check if current hypothesis meets quality criteria.
        
        Args:
            min_r_squared: Minimum R² score
            max_complexity: Maximum allowed complexity
            
        Returns:
            True if hypothesis is good enough for LLM validation
        """
        return (self.best_equation is not None and 
                self.best_r_squared >= min_r_squared and 
                self.best_complexity <= max_complexity)
    
    def get_hypothesis_summary(self) -> Dict[str, Any]:
        """Get summary of current best hypothesis.
        
        Returns:
            Dictionary with equation, metrics, and metadata
        """
        return {
            "equation": self.best_equation,
            "r_squared": self.best_r_squared,
            "complexity": self.best_complexity,
            "has_model": self.model is not None,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        if self.best_equation:
            return f"Theorist(equation={self.best_equation}, R²={self.best_r_squared:.3f})"
        return "Theorist(no hypothesis yet)"
