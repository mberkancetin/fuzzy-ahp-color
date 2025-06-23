from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .types import TFN, TrFN, Crisp, GFN, NumericType, Number


def centroid_method(fuzzy_number: TFN) -> float:
    """
    Defuzzify a triangular fuzzy number using the centroid method.
    This method returns the x-coordinate of the center of gravity of the fuzzy number.

    Parameters:
    -----------
    fuzzy_number : TFN
        The triangular fuzzy number

    Returns:
    --------
    float
        The defuzzified value
    """
    return (fuzzy_number.l + fuzzy_number.m + fuzzy_number.u) / 3.0

def centroid_method_trfn(trfn: TrFN) -> float:
    num = (trfn.d**2 + trfn.c**2 + trfn.d*trfn.c) - (trfn.a**2 + trfn.b**2 + trfn.a*trfn.b)
    den = 3 * ((trfn.d + trfn.c) - (trfn.a + trfn.b))
    return num / den if abs(den) > 1e-9 else (trfn.a + trfn.d) / 2.0

def centroid_method_gfn(gfn: GFN) -> float:
    # The centroid of a symmetric Gaussian distribution is its mean.
    return gfn.m

def graded_mean_integration(fuzzy_number: TFN) -> float:
    """
    Defuzzify a triangular fuzzy number using the graded mean integration method.
    This method gives more weight to the middle value compared to the lower and upper bounds.

    Parameters:
    -----------
    fuzzy_number : TFN
        The triangular fuzzy number

    Returns:
    --------
    float
        The defuzzified value
    """
    # The formula for triangular fuzzy numbers is (l + 4m + u) / 6
    return (fuzzy_number.l + 4 * fuzzy_number.m + fuzzy_number.u) / 6.0

def alpha_cut_method(fuzzy_number: TFN, alpha: float = 0.5) -> float:
    """
    Defuzzify a triangular fuzzy number using the alpha-cut method.
    This method takes a horizontal slice at height alpha and returns the midpoint
    of the resulting interval.

    Parameters:
    -----------
    fuzzy_number : TFN
        The triangular fuzzy number
    alpha : float
        The alpha value (0 to 1)

    Returns:
    --------
    float
        The defuzzified value
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha value must be between 0 and 1")

    # Calculate the alpha-cut interval [L_alpha, R_alpha]
    # For triangular fuzzy numbers:
    # L_alpha = l + alpha * (m - l)
    # R_alpha = u - alpha * (u - m)
    l_alpha = fuzzy_number.l + alpha * (fuzzy_number.m - fuzzy_number.l)
    r_alpha = fuzzy_number.u - alpha * (fuzzy_number.u - fuzzy_number.m)

    # Return the midpoint of the alpha-cut interval
    return (l_alpha + r_alpha) / 2.0

def weighted_average_method(fuzzy_number: TFN, weights: list = None) -> float:
    """
    Defuzzify a triangular fuzzy number using the weighted average method.
    This method applies user-defined weights to each component of the fuzzy number.

    Parameters:
    -----------
    fuzzy_number : TFN
        The triangular fuzzy number
    weights : list
        Weights for the three values (l, m, u). If None, equal weights are used.

    Returns:
    --------
    float
        The defuzzified value
    """
    # Default to equal weights if none provided
    if weights is None:
        weights = [1/3, 1/3, 1/3]

    # Validate weights
    if len(weights) != 3:
        raise ValueError("Weights must be a list of 3 values")
    if abs(sum(weights) - 1.0) > 1e-10:
        raise ValueError("Weights must sum to 1")

    # Apply weights
    return (weights[0] * fuzzy_number.l +
            weights[1] * fuzzy_number.m +
            weights[2] * fuzzy_number.u)

def normalize_crisp_weights(crisp_weights: np.ndarray) -> np.ndarray:
    """
    Normalize crisp weights to sum to 1.

    Parameters:
    -----------
    crisp_weights : np.ndarray
        Array of crisp weights

    Returns:
    --------
    np.ndarray
        Normalized crisp weights
    """
    # Check for all zeros
    if np.all(crisp_weights == 0):
        n = len(crisp_weights)
        return np.ones(n) / n  # Return equal weights if all are zero

    # Normalize to sum to 1
    weight_sum = np.sum(crisp_weights)
    if weight_sum > 0:
        return crisp_weights / weight_sum
    else:
        raise ValueError("Sum of weights is not positive, cannot normalize")

class Defuzzification:
    """
    Class to handle different defuzzification methods.
    """

    @staticmethod
    def available_methods() -> dict:
        """
        Get a dictionary containing list of available defuzzification methods.

        Returns:
        --------
        dict
            List of method names for each fuzzy or crisp number
        """
        return {
            "Crisp": None,
            "TFN": ["centroid", "graded_mean", "alpha_cut", "weighted_average", "pessimistic", "optimistic"],
            "TrFN": ["centroid", "average"],
            "GNF": ["centroid", "pessimistic_99_percent"]
        }

    @staticmethod
    def defuzzify(fuzzy_number: NumericType, method: str = "centroid", **kwargs) -> float:
        """
        Defuzzify a generic NumericType using the specified method by dispatching
        to the correct underlying algorithm based on its type.

        Parameters:
        -----------
        fuzzy_number : NumericType
            The fuzzy or crisp number (TFN, TrFN, Crisp).
        method : str
            Defuzzification method to use
        **kwargs : dict
            Additional parameters for specific methods

        Returns:
        --------
        float
            The defuzzified value
        """
        from .types import TFN, TrFN, GFN, Crisp
        if isinstance(fuzzy_number, TFN):
            if method == "centroid":
                return centroid_method(fuzzy_number)
            elif method == "graded_mean":
                return graded_mean_integration(fuzzy_number)
            elif method == "alpha_cut":
                alpha = kwargs.get("alpha", 0.5)
                return alpha_cut_method(fuzzy_number, alpha)
            elif method == "weighted_average":
                weights = kwargs.get("weights", None)
                return weighted_average_method(fuzzy_number, weights)
            elif method == "pessimistic":
                return fuzzy_number.l
            elif method == "optimistic":
                return fuzzy_number.u
            else:
                raise ValueError(f"Unknown defuzzification method: {method}")
        elif isinstance(fuzzy_number, TrFN):
            if method == "centroid":
                return centroid_method_trfn(fuzzy_number)
            elif method == "average":
                return (fuzzy_number.a + fuzzy_number.b + fuzzy_number.c + fuzzy_number.d) / 4.0
            else:
                raise ValueError(f"Method '{method}' not implemented for TrFN.")
        elif isinstance(fuzzy_number, GFN):
            if method == "centroid":
                return centroid_method_gfn(fuzzy_number)
            elif method == "pessimistic_99_percent":
                return fuzzy_number.m - 3 * fuzzy_number.sigma
            else:
                raise ValueError(f"Method '{method}' not implemented for GFN.")
        elif isinstance(fuzzy_number, Crisp):
            return fuzzy_number.value
        # Fallback for other potential types
        elif hasattr(fuzzy_number, 'defuzzify'):
            return fuzzy_number.defuzzify(method, **kwargs)
        else:
            raise TypeError(f"Unsupported type for defuzzification: {type(fuzzy_number)}")

