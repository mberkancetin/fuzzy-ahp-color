from __future__ import annotations
import numpy as np
import math
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
        "IFN": ['centroid', 'score', 'normalized_score', 'entropy', 'accuracy', 'value']
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
    if np.all(crisp_weights == 0):
        n = len(crisp_weights)
        return np.ones(n) / n

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
    # The formula for triangular fuzzy numbers is
    # (l + 4m + u) / 6
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

def weighted_average_method(fuzzy_number: TFN, weights: list = None, epsilon: float | None = None) -> float:
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
    from .config import configure_parameters
    final_epsilon = epsilon if epsilon is not None else configure_parameters.LOG_EPSILON

    if weights is None:
        weights = [1/3, 1/3, 1/3]
    if len(weights) != 3:
        raise ValueError("Weights must be a list of 3 values")
    if abs(sum(weights) - 1.0) > final_epsilon:
        raise ValueError("Weights must sum to 1")

    return (weights[0] * fuzzy_number.l +
            weights[1] * fuzzy_number.m +
            weights[2] * fuzzy_number.u)

def average_method_trfn(trfn: TrFN) -> float:
    return (trfn.a + trfn.b + trfn.c + trfn.d) / 4.0

def centroid_method_trfn(trfn: TrFN, tolerance: float | None = None) -> float:
    from .config import configure_parameters
    final_tolerance = tolerance if tolerance is not None else configure_parameters.FLOAT_TOLERANCE

    num = (trfn.d**2 + trfn.c**2 + trfn.d*trfn.c) - (trfn.a**2 + trfn.b**2 + trfn.a*trfn.b)
    den = 3 * ((trfn.d + trfn.c) - (trfn.a + trfn.b))
    return num / den if abs(den) > final_tolerance else (trfn.a + trfn.d) / 2.0

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

def chen_tan_scoring_ifn(ifn: IFN) -> float:
    """
    Calculates a normalized score for an IFN on a [0, 1] scale.
    Based on Chen & Tan (1994), this is suitable for using IFNs as
    weights or performance values in aggregation, as it's always positive.
    Formula: S = (μ + 1 - ν) / 2
    """
    return (ifn.mu + 1.0 - ifn.nu) / 2.0

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

def piecewise_score_method_ifn(ifn: IFN, lambda_param: float = 2.0) -> float:
    """
    An enhanced score function for IFNs that incorporates the hesitancy degree.
    Based on Yang et al. (2023) as cited in Zhou & Chen (2025, Eq. 18).

    Args:
        ifn: The Intuitionistic Fuzzy Number.
        lambda_param: Controls the influence of the hesitancy degree. Must be >= 2.
    """
    if lambda_param < 2:
        raise ValueError("Lambda parameter must be >= 2.")

    mu, nu, h = ifn.mu, ifn.nu, ifn.hesitancy()
    score_part = mu - nu

    if h == 0:
        return score_part

    term = 1 + (lambda_param**abs(mu - nu)) / (10**lambda_param)
    log_part = h * math.log(term, lambda_param)

    return score_part + log_part if mu >= nu else score_part - log_part

def modal_transform_ifn(ifn: IFN, alpha: float = 0.5, beta: float = 0.0, **kwargs) -> float:
    """
    Defuzzifies an IFN after applying an Extended Modal Operator (F_α,β).

    This method, based on Marinov (2022) after Atanassov, allows for expressing
    optimism or pessimism by distributing the hesitation degree (π) into the
    membership (μ) and non-membership (ν) degrees before final scoring.

    Args:
        ifn (IFN): The Intuitionistic Fuzzy Number to defuzzify.
        alpha (float, optional): The proportion of hesitation to convert into
                                 membership (optimism). Must be in [0, 1].
                                 Defaults to 0.5.
        beta (float, optional): The proportion of hesitation to convert into
                                non-membership (pessimism). Must be in [0, 1].
                                Defaults to 0.0.

    Returns:
        float: The final crisp score, typically on a [0, 1] scale.
    """
    if not (0 <= alpha <= 1 and 0 <= beta <= 1):
        raise ValueError("alpha and beta must be between 0 and 1.")
    if alpha + beta > 1:
        raise ValueError("The sum of alpha and beta must not exceed 1.")

    pi = ifn.pi

    # Apply the F_α,β transformation
    new_mu = ifn.mu + alpha * pi
    new_nu = ifn.nu + beta * pi

    # After the transformation, the hesitation of the new IFN is reduced.
    # S = (μ' + 1 - ν') / 2
    final_score = (new_mu + 1.0 - new_nu) / 2.0

    return final_score

def xu_yager_scoring_ifn(ifn: IFN, alpha: float = 0.5) -> float:
    """
    Calculates a normalized score that incorporates the hesitation degree (pi).
    Based on Xu and Yager (2006), this allows for modeling different attitudes
    towards uncertainty.

    Formula: S = mu + alpha * pi

    Args:
        ifn (IFN): The Intuitionistic Fuzzy Number to defuzzify.
        alpha (float, optional): The attitudinal parameter (0=pessimistic,
                                 0.5=neutral, 1=optimistic). Defaults to 0.5.

    Returns:
        A crisp score on a [0, 1] scale.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha parameter must be between 0 and 1.")

    return ifn.mu + alpha * ifn.pi

# ==============================================================================
# 2. REGISTRATION
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
IFN.register_defuzzify_method('normalized_score', chen_tan_scoring_ifn)
IFN.register_defuzzify_method('centroid', xu_yager_scoring_ifn) # 'centroid' is an alias for 'xu_yager'
IFN.register_defuzzify_method('xu_yager', xu_yager_scoring_ifn) 
IFN.register_defuzzify_method('accuracy', accuracy_method_ifn)
IFN.register_defuzzify_method('value', value_method_ifn)
IFN.register_defuzzify_method('entropy', entropy_method_ifn)
IFN.register_defuzzify_method('piecewise_score', piecewise_score_method_ifn)
IFN.register_defuzzify_method('modal_transform', modal_transform_ifn)
