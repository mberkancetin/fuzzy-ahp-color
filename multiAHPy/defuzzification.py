from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from .types import TFN, TrFN, IFN, IT2TrFN, Crisp, GFN, NumericType, Number


def available_methods() -> dict:
    """
    Get a dictionary containing list of available defuzzification methods.

    Returns:
    --------
    dict
        List of method names for each fuzzy or crisp number
    """
    return {
        "Crisp": ["centroid", "value"], # to keep integrity
        "TFN": ["centroid", "graded_mean", "alpha_cut", "weighted_average", "pessimistic", "optimistic"],
        "TrFN": ["centroid", "average"],
        "GNF": ["centroid", "pessimistic_99_percent"],
        "IFN": ['centroid', 'score', 'entropy', 'accuracy', 'value']
    }

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

def value_method_crisp(crisp: Crisp) -> float:
    return crisp.value

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

def graded_mean_integration(fuzzy_number: TFN) -> float:
    """
    Defuzzify a triangular fuzzy number using the graded mean integration method.
    This method gives more weight to the middle value compared to the lower and upper bounds.

    .. note::
            For TFNs, 'graded_mean' (l+4m+u)/6 is often preferred as it is a
            well-regarded method (e.g., Yager's approach) that considers all
            points of the fuzzy number. 'centroid' (l+m+u)/3 is simpler.

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

def average_method_trfn(trfn: TrFN) -> float:
    return (trfn.a + trfn.b + trfn.c + trfn.d) / 4.0

def centroid_method_trfn(trfn: TrFN) -> float:
    num = (trfn.d**2 + trfn.c**2 + trfn.d*trfn.c) - (trfn.a**2 + trfn.b**2 + trfn.a*trfn.b)
    den = 3 * ((trfn.d + trfn.c) - (trfn.a + trfn.b))
    return num / den if abs(den) > 1e-9 else (trfn.a + trfn.d) / 2.0

def centroid_method_it2trfn(fuzzy_number: IT2TrFN) -> float:
    centroid_umf = fuzzy_number.umf.defuzzify(method='centroid')
    centroid_lmf = fuzzy_number.lmf.defuzzify(method='centroid')
    return (centroid_umf + centroid_lmf) / 2.0

def pessimistic_method_gfn(gfn: GFN) -> float:
    return gfn.m - 3 * gfn.sigma

def score_method_ifn(ifn: IFN) -> float:
    """
    Calculates the Score Function (S = μ - ν). This is the most common
    method for ranking IFNs. It represents the net degree of membership
    over non-membership. A value of 1 is best, -1 is worst.
    """
    return ifn.mu - ifn.nu

def accuracy_method_ifn(ifn: IFN) -> float:
    """
    Calculates the Accuracy Function (H = μ + ν). This value represents the
    degree of certainty or information about the judgment. It is typically
    not used for primary ranking, but as a tie-breaker when two IFNs have
    the same score.
    """
    return ifn.mu + ifn.nu

def value_method_ifn(ifn: IFN) -> float:
    """
    Calculates a value by considering the hesitation degree (π) as potential
    membership, distributed proportionally to the existing membership (μ).
    Formula: V = μ + π*μ. This can be seen as a slightly more optimistic
    measure than the simple score function.
    """
    return ifn.mu + (ifn.pi * ifn.mu)

def entropy_method_ifn(ifn: IFN) -> float:
    """
    Calculates the Intuitionistic Fuzzy Entropy. This does not represent
    the value of the number, but its degree of fuzziness or uncertainty.
    A score of 0 means no fuzziness (crisp), 1 means maximum fuzziness.
    Based on Burillo and Bustince (1996).
    """
    return 1.0 - abs(ifn.mu - ifn.nu)


# ==============================================================================
# 2. REGISTRATION (The "Wiring")
# ==============================================================================

Crisp.register_defuzzify_method('centroid', value_method_crisp)
Crisp.register_defuzzify_method('value', value_method_crisp)

TFN.register_defuzzify_method('centroid', centroid_method)
TFN.register_defuzzify_method('graded_mean', graded_mean_integration)
TFN.register_defuzzify_method('alpha_cut', alpha_cut_method)
TFN.register_defuzzify_method('pessimistic', lambda tfn: tfn.l)
TFN.register_defuzzify_method('optimistic', lambda tfn: tfn.u)

TrFN.register_defuzzify_method('centroid', centroid_method_trfn)
TrFN.register_defuzzify_method('average', average_method_trfn)

IT2TrFN.register_defuzzify_method('centroid_average', centroid_method_it2trfn)
IT2TrFN.register_defuzzify_method('centroid', centroid_method_it2trfn) # 'centroid' is an alias for 'centroid_average'

GFN.register_defuzzify_method('centroid', lambda gfn: gfn.m)
GFN.register_defuzzify_method('pessimistic_99_percent', pessimistic_method_gfn)

# --- Academic Note on IFN Defuzzification ---
# Defuzzification of IFNs aims to convert the (μ, ν) pair into a single
# crisp value for ranking. Different methods exist based on how the
# hesitation degree (π = 1 - μ - ν) is handled.
IFN.register_defuzzify_method('score', score_method_ifn)
IFN.register_defuzzify_method('centroid', score_method_ifn) # 'centroid' is an alias for 'score'
IFN.register_defuzzify_method('accuracy', accuracy_method_ifn)
IFN.register_defuzzify_method('value', value_method_ifn)
IFN.register_defuzzify_method('entropy', entropy_method_ifn)
