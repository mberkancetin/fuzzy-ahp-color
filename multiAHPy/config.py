from typing import Dict, Any

class Configuration:
    """
    A singleton-like class to hold all configurable parameters for the multiAHPy library.

    Users can modify these attributes directly to customize the behavior of
    consistency checks, aggregation methods, and other algorithms.

    Example:
    >>> from multiAHPy.config import configure_parameters
    >>> # Use a different set of RI values for a specific study
    >>> configure_parameters.SAATY_RI_VALUES = {1: 0, 2: 0, 3: 0.58, ...}
    >>> # Set a stricter GCI threshold
    >>> configure_parameters.GCI_THRESHOLDS = {3: 0.25, 4: 0.30, 'default': 0.32}
    """

    def __init__(self):
        self.reset_to_defaults()

    def reset_to_defaults(self):
        """Resets all configuration parameters to their original default values."""

        # --- Consistency Parameters (from consistency.py) ---

        # Saaty's Random Consistency Index (RI) values
        # Source: Saaty, T. L. (2008)
        self.SAATY_RI_VALUES: Dict[int, float] = {
            1: 0.00, 2: 0.00, 3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35,
            8: 1.40, 9: 1.45, 10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59,
            'default': 1.60 # Default for n > 15
        }

        # Default Saaty's CR threshold
        self.DEFAULT_SAATY_CR_THRESHOLD: float = 0.1


        # Geometric Consistency Index (GCI) thresholds
        # Source: Aguarón & Moreno-Jiménez (2003)
        self.GCI_THRESHOLDS: Dict[int | str, float] = {
            3: 0.31,
            4: 0.35,
            'default': 0.37 # Default for n > 4
        }

        # --- Fuzzy Scale Parameters (from matix_builder.py) ---

        # Predefined TFN scales
        self.FUZZY_TFN_SCALES: Dict[str, Dict[int, tuple]] = {
            "linear": {1: (1, 1, 1), 2: (1, 2, 3), 3: (2, 3, 4), 4: (3, 4, 5), 5: (4, 5, 6), 6: (5, 6, 7), 7: (6, 7, 8), 8: (7, 8, 9), 9: (8, 9, 9)},
            "saaty_original": {1: (1, 1, 1), 2: (1, 2, 3), 3: (2, 3, 4), 4: (3, 4, 5), 5: (4, 5, 6), 6: (5, 6, 7), 7: (6, 7, 8), 8: (7, 8, 9), 9: (9, 9, 9)},
            "wide": {1: (1, 1, 3), 2: (1, 2, 4), 3: (2, 3, 5), 4: (3, 4, 6), 5: (4, 5, 7), 6: (5, 6, 8), 7: (6, 7, 9), 8: (7, 8, 9), 9: (8, 9, 9)},
            "narrow": {1: (1, 1, 1), 2: (1.5, 2, 2.5), 3: (2.5, 3, 3.5), 4: (3.5, 4, 4.5), 5: (4.5, 5, 5.5), 6: (5.5, 6, 6.5), 7: (6.5, 7, 7.5), 8: (7.5, 8, 8.5), 9: (8.5, 9, 9)}
        }

        # Predefined IFN scales
        # Source: Nguyen (2019)
        self.FUZZY_IFN_SCALES: Dict[int, tuple] = {
            "nyugen": {1: (0.50, 0.40), 2: (0.55, 0.35), 3: (0.60, 0.30), 4: (0.65, 0.25), 5: (0.70, 0.20), 6: (0.75, 0.15), 7: (0.80, 0.10), 8: (0.90, 0.05), 9: (1.00, 0.00)}
        }

        # --- General Numerical Parameters ---

        # Small tolerance value for float comparisons, reciprocity checks, etc.
        self.FLOAT_TOLERANCE: float = 1e-9

        # Epsilon value to prevent log(0) errors
        self.LOG_EPSILON: float = 1e-10

    def register_tfn_scale(self, name: str, scale_definition: Dict[int, tuple]):
        """Registers a new TFN conversion scale to the configuration."""
        if name in self.FUZZY_TFN_SCALES:
            print(f"Warning: Overwriting TFN scale '{name}'")
        self.FUZZY_TFN_SCALES[name] = scale_definition

    def register_ifn_scale(self, name: str, scale_definition: Dict[int, tuple]):
        """Registers a new IFN conversion scale to the configuration."""
        if name in self.FUZZY_IFN_SCALES:
            print(f"Warning: Overwriting IFN scale '{name}'")
        self.FUZZY_IFN_SCALES[name] = scale_definition

configure_parameters = Configuration()
