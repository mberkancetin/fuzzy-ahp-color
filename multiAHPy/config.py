from typing import Dict, Tuple, Callable
import numpy as np


ScaleFunction = Callable[[float], Tuple[float, ...]]


def linear_tfn_scale(value: float) -> Tuple[float, float, float]:
    """
    A TFN scale where the spread is consistently +/- 1 from the value,
    with bounds clipped to the [1, 9] Saaty scale.
    """
    v = float(value)
    m = np.clip(v, 1, 9)
    l = m - 1
    u = m + 1
    return (max(1.0, l), m, min(9.0, u))

def wide_tfn_scale(value: float) -> Tuple[float, float, float]:
    """
    A TFN scale with a wider, asymmetric spread of -1, +2.
    """
    v = float(value)
    m = np.clip(v, 1, 9)
    l = m - 1
    u = m + 2
    return (max(1.0, l), m, min(9.0, u))

def narrow_tfn_scale(value: float) -> Tuple[float, float, float]:
    """
    A TFN scale with a consistent, narrow spread of +/- 0.5.
    """
    v = float(value)
    m = np.clip(v, 1, 9)
    l = m - 0.5
    u = m + 0.5
    return (max(1.0, l), m, min(9.0, u))

def asymptotic_tfn_scale(value: float) -> Tuple[float, float, float]:
    """
    A sophisticated TFN scale that behaves linearly within [1, 9] and
    asymptotically approaches a crisp '9' for values > 9.
    """
    v = float(value)
    if v <= 1:
        return (1.0, 1.0, 1.0)
    if v <= 9:
        m = v
        l = v - 1
        return (max(1.0, l), m, min(9.0, v + 1))
    else: # v > 9
        excess = v - 9.0
        # Lower bound approaches 9 from 8 as excess grows
        l = 8.0 + (1.0 - (1.0 / (1.0 + excess)))
        return (l, 9.0, 9.0)



def _interpolate_from_dict(value: float, scale_dict: Dict[int, tuple]) -> Tuple[float, float]:
    """A generic helper to interpolate (mu, nu) pairs from a dictionary scale."""
    value = float(value)
    max_key = max(scale_dict.keys())
    value = np.clip(value, 1, max_key)

    if value == int(value) and int(value) in scale_dict:
        return scale_dict[int(value)]

    lower_key = int(np.floor(value))
    upper_key = int(np.ceil(value))

    # Handle edge case where value is exactly on an integer
    if lower_key == upper_key:
        return scale_dict[lower_key]

    # Handle cases where one key might not exist (e.g., in a 5-level scale for value 2.5)
    if lower_key not in scale_dict or upper_key not in scale_dict:
        # Fallback: snap to the nearest valid key
        closest_key = min(scale_dict.keys(), key=lambda k: abs(k - value))
        return scale_dict[closest_key]

    lower_params = scale_dict[lower_key]
    upper_params = scale_dict[upper_key]

    weight = value - lower_key

    mu = lower_params[0] * (1 - weight) + upper_params[0] * weight
    nu = lower_params[1] * (1 - weight) + upper_params[1] * weight

    return (mu, nu)

# ACADEMIC CITATION:
# Adapted from Büyüközkan, G., & Göçer, F. (2018).
# "An intuitionistic fuzzy MCDM approach for effective hazardous waste management."
# Scale for Intuitionistic Fuzzy Numbers.
ACADEMIC_IFN_SCALE = {
    1: (0.50, 0.40),
    2: (0.55, 0.35),
    3: (0.60, 0.30),
    4: (0.65, 0.25),
    5: (0.70, 0.20),
    6: (0.75, 0.15),
    7: (0.80, 0.10),
    8: (0.85, 0.05),
    9: (0.90, 0.05)  # Note: usually sum is < 1 to allow hesitation
}

def nguyen_9_level_ifn_scale(value: float) -> Tuple[float, float]:
    """Functional IFN scale based on Nguyen (2019) with interpolation."""
    scale_dict = {
        1: (0.50, 0.40), 2: (0.55, 0.35), 3: (0.60, 0.30), 4: (0.65, 0.25),
        5: (0.70, 0.20), 6: (0.75, 0.15), 7: (0.80, 0.10), 8: (0.90, 0.05), 9: (1.00, 0.00)
    }
    return _interpolate_from_dict(value, scale_dict)

def buyukozkan_9_level_ifn_scale(value: float) -> Tuple[float, float]:
    """Functional IFN scale based on Büyüközkan (2016) with interpolation."""
    scale_dict = ACADEMIC_IFN_SCALE
    return _interpolate_from_dict(value, scale_dict)

def dymova_9_level_ifn_scale(value: float) -> Tuple[float, float]:
    """Functional IFN scale based on Dymova & Sevastjanov with interpolation."""
    scale_dict = {
        1: (0.10, 0.90), 2: (0.20, 0.75), 3: (0.35, 0.60), 4: (0.45, 0.50),
        5: (0.50, 0.45), 6: (0.60, 0.35), 7: (0.75, 0.20), 8: (0.85, 0.10), 9: (0.90, 0.10)
    }
    return _interpolate_from_dict(value, scale_dict)

def chen_tan_5_level_ifn_scale(value: float) -> Tuple[float, float]:
    """Functional 5-level IFN scale with interpolation."""
    scale_dict = {
        1: (0.1, 0.8), 2: (0.3, 0.6), 3: (0.5, 0.4), 4: (0.7, 0.2), 5: (0.9, 0.0)
    }
    return _interpolate_from_dict(value, scale_dict)

def symmetrical_log_base_scale(value: float, scale_base: int = 9)-> Tuple[float, float]:

    if value <= 0:
            raise ValueError("Saaty-scale value must be positive.")

    pi = 0.1

    log_val = np.log(value) / np.log(scale_base)
    linguistic_anchor = (log_val + 1) / 2.0

    mu = (1 - pi) * linguistic_anchor
    nu = 1 - mu - pi

    mu = np.clip(mu, 0, 1)
    nu = np.clip(nu, 0, 1)
    if mu + nu > 1.0:
        nu = 1.0 - mu

    return (mu, nu)

RI_Approximation_Func = Callable[[int, int, Dict[int, float]], float]


def default_ri_approximation(n: int, m: int, ri_table: Dict[int, float]) -> float:
    """
    Calculates the generalized RI using the linear approximation from
    Ágoston & Csató (2022), Equation (3).

    Args:
        n: Matrix size.
        m: Number of missing pairs above the diagonal.
        ri_table: The dictionary of standard RI values for complete matrices.
    """
    ri_complete = ri_table.get(n, ri_table.get('default', 1.6))
    denominator = (n - 1) * (n - 2)
    if denominator <= 0:
        return 0.0 # Not applicable for n<=2

    # The formula from the paper is:
    # RI_n,m ≈ (1 - 2m / ((n-1)(n-2))) * RI_n,0
    approximation = (1 - (2 * m) / denominator) * ri_complete
    return max(0, approximation)


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

        # Generalized RI for INCOMPLETE matrices (RI_n,m)
        # Source: Ágoston & Csató (2022), Table 2
        # Keys are tuples of (matrix_size, num_missing_above_diagonal)
        self.GENERALIZED_RI_VALUES: Dict[Tuple[int, int], float] = {
            # n=4
            (4, 0): 0.884, (4, 1): 0.583, (4, 2): 0.306, (4, 3): 0.053,
            # n=5
            (5, 0): 1.109, (5, 1): 0.925, (5, 2): 0.739, (5, 3): 0.557,
            (5, 4): 0.379, (5, 5): 0.212, (5, 6): 0.059,
            # n=6
            (6, 0): 1.249, (6, 1): 1.128, (6, 2): 1.007, (6, 3): 0.883,
            (6, 4): 0.758, (6, 5): 0.634, (6, 6): 0.510, (6, 7): 0.389,
            (6, 8): 0.271, (6, 9): 0.161,
            # n=7
            (7, 0): 1.341, (7, 1): 1.256
        }

        # Linear approximation formula for RI_n,m
        # Source: Ágoston & Csató (2022), Equation (3)
        # This is a fallback for when a value is not in the table.
        self.USE_RI_APPROXIMATION_FALLBACK = True

        # Default Saaty's CR threshold
        self.DEFAULT_SAATY_CR_THRESHOLD: float = 0.1

        # Geometric Consistency Index (GCI) thresholds
        # Source: Aguarón & Moreno-Jiménez (2003)
        self.GCI_THRESHOLDS: Dict[int | str, float] = {
            3: 0.31,
            4: 0.35,
            'default': 0.37 # Default for n > 4
        }



        self.FUZZY_TFN_SCALES_FUNCTIONS: Dict[str, ScaleFunction] = {
            "linear": linear_tfn_scale,
            "wide": wide_tfn_scale,
            "narrow": narrow_tfn_scale,
            "asymptotic": asymptotic_tfn_scale
        }

        self.FUZZY_IFN_SCALES_FUNCTIONS: Dict[str, ScaleFunction] = {
            "nguyen_9_level": nguyen_9_level_ifn_scale,
            "buyukozkan_9_level": buyukozkan_9_level_ifn_scale,
            "dymova_9_level": dymova_9_level_ifn_scale,
            "chen_tan_5_level": chen_tan_5_level_ifn_scale,
            "symmetrical_log_base": symmetrical_log_base_scale
        }

        # --- Deprecated / Legacy Dictionary Scales ---

        self.FUZZY_TFN_SCALES: Dict[str, Dict[int, tuple]] = {
            "linear": {
                1: (1, 1, 1), 2: (1, 2, 3), 3: (2, 3, 4), 4: (3, 4, 5),
                5: (4, 5, 6), 6: (5, 6, 7), 7: (6, 7, 8), 8: (7, 8, 9), 9: (8, 9, 9)
            },
            "saaty_original": {
                1: (1, 1, 1), 2: (1, 2, 3), 3: (2, 3, 4), 4: (3, 4, 5),
                5: (4, 5, 6), 6: (5, 6, 7), 7: (6, 7, 8), 8: (7, 8, 9), 9: (9, 9, 9)
            },
            "wide": {
                1: (1, 1, 3), 2: (1, 2, 4), 3: (2, 3, 5), 4: (3, 4, 6),
                5: (4, 5, 7), 6: (5, 6, 8), 7: (6, 7, 9), 8: (7, 8, 9), 9: (8, 9, 9)
            },
            "narrow": {
                1: (1, 1, 1), 2: (1.5, 2, 2.5), 3: (2.5, 3, 3.5), 4: (3.5, 4, 4.5),
                5: (4.5, 5, 5.5), 6: (5.5, 6, 6.5), 7: (6.5, 7, 7.5), 8: (7.5, 8, 8.5), 9: (8.5, 9, 9)
            }
        }

        self.FUZZY_IFN_SCALES: Dict[str, Dict[int, tuple]] = {
            "nguyen_9_level": {
                1: (0.50, 0.40), 2: (0.55, 0.35), 3: (0.60, 0.30), 4: (0.65, 0.25),
                5: (0.70, 0.20), 6: (0.75, 0.15), 7: (0.80, 0.10), 8: (0.90, 0.05), 9: (1.00, 0.00)
            },
            "buyukozkan_9_level": ACADEMIC_IFN_SCALE,
            "dymova_9_level": {
                1: (0.10, 0.90), 2: (0.20, 0.75), 3: (0.35, 0.60), 4: (0.45, 0.50),
                5: (0.50, 0.45), 6: (0.60, 0.35), 7: (0.75, 0.20), 8: (0.85, 0.10), 9: (0.90, 0.10)
            },
            "chen_tan_5_level": {
                1: (0.1, 0.8), 2: (0.3, 0.6), 3: (0.5, 0.4), 4: (0.7, 0.2), 5: (0.9, 0.0)
            }
        }



        self.FUZZY_IFN_RATING_SCALES: Dict[str, Dict[int, tuple]] = {
             "boran_rating_7_level": {
                1: (0.90, 0.10), 2: (0.75, 0.20), 3: (0.60, 0.35), 4: (0.50, 0.45),
                5: (0.40, 0.50), 6: (0.25, 0.70), 7: (0.10, 0.90)
            }
        }

        # --- General Numerical Parameters ---

        # Small tolerance value for float comparisons, reciprocity checks, etc.
        self.FLOAT_TOLERANCE: float = 1e-9

        # Epsilon value to prevent log(0) errors
        self.LOG_EPSILON: float = 1e-10

        self.RI_APPROXIMATION_FUNC: RI_Approximation_Func = default_ri_approximation


    def register_tfn_scale_function(self, name: str, func: ScaleFunction):
        """Registers a new TFN scale function."""
        if name in self.FUZZY_TFN_SCALES_FUNCTIONS:
            print(f"Warning: Overwriting TFN scale function '{name}'")
        self.FUZZY_TFN_SCALES_FUNCTIONS[name] = func

    def register_ifn_scale_function(self, name: str, func: ScaleFunction):
        """Registers a new IFN scale function."""
        if name in self.FUZZY_IFN_SCALES_FUNCTIONS:
            print(f"Warning: Overwriting IFN scale function '{name}'")
        self.FUZZY_IFN_SCALES_FUNCTIONS[name] = func

    # --- Deprecated / Legacy Dictionary Scales ---

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



class ConfigurationContextManager:
    """
    A context manager to temporarily change configuration parameters.

    Usage:
    >>> with ConfigurationContextManager(DEFAULT_SAATY_CR_THRESHOLD=0.05):
    >>>     # Code block runs with CR threshold set to 0.05
    >>>     ...
    >>> # CR threshold reverts to its original value outside the block
    """
    def __init__(self, **kwargs):
        self.changes = kwargs
        self.original_values = {}

    def __enter__(self):
        for key, value in self.changes.items():
            if not hasattr(configure_parameters, key):
                raise AttributeError(f"Configuration object has no attribute '{key}'")
            self.original_values[key] = getattr(configure_parameters, key)
            setattr(configure_parameters, key, value)
        return configure_parameters

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.original_values.items():
            setattr(configure_parameters, key, value)
