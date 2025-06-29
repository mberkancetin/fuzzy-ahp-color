from __future__ import annotations
from typing import List, Type, Dict, Any, TYPE_CHECKING
import numpy as np
import math
import warnings

if TYPE_CHECKING:
    from .types import NumericType, Number, TFN, Crisp, IFN, IT2TrFN

try:
    from scipy.optimize import minimize
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

def _check_scipy_availability():
    if not _SCIPY_AVAILABLE:
        raise ImportError("Fuzzy Programming methods require the 'scipy' library. "
                          "Please install it using: pip install scipy")


# ==============================================================================
# 1. REGISTRY FOR CUSTOMIZATION
# ==============================================================================

WEIGHT_DERIVATION_REGISTRY = {}

def register_weight_method(number_type_name: str, method_name: str):
    """A decorator to register a new weight derivation method."""
    def decorator(func):
        if (number_type_name, method_name) in WEIGHT_DERIVATION_REGISTRY:
            print(f"Warning: Overwriting existing weight method for ({number_type_name}, {method_name})")
        WEIGHT_DERIVATION_REGISTRY[(number_type_name, method_name)] = func
        return func
    return decorator


# ==============================================================================
# 2. GENERIC & CLASSIC AHP ALGORITHMS
# ==============================================================================

@register_weight_method('TFN', 'geometric_mean')
@register_weight_method('TrFN', 'geometric_mean')
@register_weight_method('GFN', 'geometric_mean')
@register_weight_method('IFN', 'geometric_mean')
@register_weight_method('IT2TrFN', 'geometric_mean')
@register_weight_method('Crisp', 'geometric_mean')
def geometric_mean_method(matrix: np.ndarray, number_type: Type[Number], consistency_method: str = "centroid") -> List[Number]:
    """
    Derives weights using the fuzzy geometric mean method.

    .. note::
        **Academic Note:** This method is a direct extension of the geometric mean
        used in classical AHP. It is generally considered a robust and stable method
        that correctly handles the ratio-scale nature of AHP judgments. It is often
        recommended for its mathematical consistency and reliable results.

    .. note::
        **Academic Note:** This is one valid approach when directly applied
        to an aggregated IFN matrix; however, another common approach in the
        literature involves calculating crisp weights for each expert first,
        then aggregating those priorities using a method like IFWA
        (see the 'aggregation' module).

    Args:
        matrix: The comparison matrix of shape (n, n).
        number_type: The class of the number type being used (e.g., Crisp, TFN).
        consistency_method: Defuzzification method to use

    Returns:
        A list of derived weights of the specified number_type.
    """
    n = matrix.shape[0]

    row_geo_means = []
    for i in range(n):
        row_product = number_type.multiplicative_identity()
        for cell in matrix[i, :]:
            row_product = row_product * cell
        row_geo_means.append(row_product.power(1.0 / n))

    total_sum = sum(row_geo_means, number_type.neutral_element())
    if abs(total_sum.defuzzify(method=consistency_method)) < 1e-9:
        return [number_type.neutral_element() for _ in range(n)]

    sum_inverse = total_sum.inverse()
    weights = [geo_mean * sum_inverse for geo_mean in row_geo_means]
    return weights

@register_weight_method('Crisp', 'eigenvector')
def eigenvector_method(matrix: np.ndarray, number_type: Type[Crisp], **kwargs) -> List[Number]:
    """
    Derives weights using the principal right eigenvector of the matrix.
    This implementation is for CRISP matrices only.

    Args:
        matrix: A crisp comparison matrix.
        number_type: Must be the Crisp class.

    Returns:
        A list of crisp weights.
    """
    if number_type.__name__ != 'Crisp':
        raise TypeError("Standard eigenvector method is only applicable to crisp matrices.")

    n = matrix.shape[0]
    crisp_matrix = np.array([[cell.value for cell in row] for row in matrix])

    eigenvalues, eigenvectors = np.linalg.eig(crisp_matrix)
    max_eig_index = np.argmax(eigenvalues)
    weights = np.real(eigenvectors[:, max_eig_index])
    normalized_weights = weights / np.sum(weights)
    return [number_type(w) for w in normalized_weights]


# ==============================================================================
# 3. FUZZY-SPECIFIC AHP ALGORITHMS
# ==============================================================================

@register_weight_method('TFN', 'extent_analysis')
@register_weight_method('TrFN', 'extent_analysis')
def extent_analysis_method(matrix: np.ndarray, number_type: Type[TFN]) -> Dict[str, Any]:
    """
    Implements Chang's (1996) extent analysis method for deriving weights from
    TFN matrices.

    .. note::
        **Academic Note:** Extent Analysis is one of the most cited methods in
        Fuzzy AHP literature. However, it has been subject to academic debate.
        Its primary output is a vector of crisp weights derived from the "minimum
        degree of possibility." Critics have pointed out that this can sometimes
        result in a zero weight for a criterion that should not be zero, and the
        method's mathematical properties differ from traditional AHP. It is
        presented here due to its historical significance and widespread use.

    .. warning::
        This method is widely cited but has been academically criticized for
        several issues, including the potential to assign a zero weight to a
        non-zero criterion and its mathematical properties differing from
        classical AHP. Use with caution and consider more robust methods
        like 'geometric_mean'. See Liu, Y. et al. (2020) for a discussion.

    Returns:
        A dictionary containing weights, crisp_weights, possibility_matrix, etc.
    """
    warnings.warn(
        "Extent Analysis (EAM) is a 'problematic' method according to recent reviews (e.g., Liu et al., 2020). It may produce zero weights incorrectly. Consider using 'geometric_mean' for more robust results.",
        UserWarning
    )

    n = matrix.shape[0]
    if not hasattr(matrix[0,0], 'possibility_degree'):
        raise TypeError("Extent analysis requires TFNs with 'possibility_degree' method.")

    # Step 1: Calculate fuzzy synthetic extent values
    row_sums = [np.sum(matrix[i, :]) for i in range(n)]
    total_sum = np.sum(row_sums)
    inverse_total = total_sum.inverse()
    synthetic_extents = [rs * inverse_total for rs in row_sums]

    # Step 2: Calculate the degree of possibility matrix (V matrix)
    V_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i, n):
            V_matrix[i, j] = synthetic_extents[i].possibility_degree(synthetic_extents[j])
            V_matrix[j, i] = synthetic_extents[j].possibility_degree(synthetic_extents[i])

    # Step 3: Calculate the minimum degree of possibility vector
    min_degrees = np.array([min(V_matrix[i, j] for j in range(n) if i != j) for i in range(n)])

    # Step 4: Normalize to get crisp weights
    weights_sum = np.sum(min_degrees)
    crisp_weights = min_degrees / weights_sum if weights_sum > 0 else np.full(n, 1/n)

    # Represent final weights as TFNs
    fuzzy_weights = [number_type.from_crisp(w) for w in crisp_weights]

    return {
        "weights": fuzzy_weights,
        "crisp_weights": crisp_weights,
        "possibility_matrix": V_matrix,
        "min_degrees": min_degrees
    }

@register_weight_method('TFN', 'llsm')
@register_weight_method('TrFN', 'llsm')
def fuzzy_llsm_method(matrix: np.ndarray, number_type: Type[Number], components: List[str]) -> List[Number]:
    """
    Derives weights using a generic Fuzzy Logarithmic Least Squares Method (LLSM).
    This method is applied component-wise to a fuzzy number.

    Args:
        matrix: The fuzzy comparison matrix.
        number_type: The class of the fuzzy number (e.g., TFN, TrFN).
        components: A list of attribute names for the components (e.g., ['l', 'm', 'u']).

    Returns:
        A list of derived fuzzy weights.
    """
    n = matrix.shape[0]

    # Helper function to apply LLSM to a single crisp component matrix
    def llsm_component(component_matrix: np.ndarray) -> np.ndarray:
        component_matrix = np.where(component_matrix == 0, 1e-10, component_matrix)
        log_matrix = np.log(component_matrix)
        # The original formula uses geometric mean of columns, which is equivalent
        # to the arithmetic mean of the log of rows.
        row_means = np.mean(log_matrix, axis=1)
        weights = np.exp(row_means)
        return weights / np.sum(weights)

    # Calculate weights for each component
    component_weights = {}
    for comp_name in components:
        comp_matrix = np.array([[getattr(cell, comp_name) for cell in row] for row in matrix])
        component_weights[comp_name] = llsm_component(comp_matrix)

    fuzzy_weights = []
    for i in range(n):
        weight_params = [component_weights[comp_name][i] for comp_name in components]

        # Ensure the parameters are in the correct order for the constructor (l<=m<=u)
        sorted_params = sorted(weight_params)

        # Instantiate the fuzzy number object
        fuzzy_weights.append(number_type(*sorted_params))

    return fuzzy_weights

@register_weight_method('TFN', 'lambda_max')
@register_weight_method('TrFN', 'lambda_max')
def lambda_max_method(matrix: np.ndarray, number_type: Type[Number]) -> List[Number]:
    """
    Derives fuzzy weights using the Lambda-max method by Csutora and Buckley (2001).
    This method fuzzifies Saaty's eigenvector method using alpha-cuts.

    .. note::
        **Academic Note:** This method directly extends the core concept of
        classical AHP (finding the principal eigenvector) to the fuzzy domain.
        It tends to produce fuzzy weights with less fuzziness (a smaller spread)
        than the geometric mean method.
    """
    n = matrix.shape[0]

    # Check if the number type supports alpha-cuts
    if not hasattr(matrix[0,0], 'alpha_cut') or matrix[0,0].alpha_cut(0.5) is NotImplemented:
        raise TypeError(f"Lambda-max method requires number types that support alpha-cuts (e.g., TFN, TrFN).")

    # Step 1: Get the middle weights (alpha = 1.0)
    # The alpha-cut at alpha=1 is just the middle point (m for TFN)
    alpha_1_cut = np.array([[cell.alpha_cut(1.0)[0] for cell in row] for row in matrix])
    # Wrap in Crisp objects to use our existing eigenvector method
    crisp_matrix_m = np.array([[Crisp(val) for val in row] for row in alpha_1_cut], dtype=object)
    weights_m_crisp = eigenvector_method(crisp_matrix_m, Crisp)
    middle_weights = np.array([w.value for w in weights_m_crisp])

    # Step 2: Get the lower bound weights (alpha = 0.0)
    # The alpha-cut at alpha=0 is the full interval [l, u]
    alpha_0_cut = np.array([[cell.alpha_cut(0.0) for cell in row] for row in matrix])
    # We need the matrix of lower bounds
    crisp_matrix_l = np.array([[interval[0] for interval in row] for row in alpha_0_cut])
    crisp_matrix_l = np.array([[Crisp(val) for val in row] for row in crisp_matrix_l], dtype=object)
    weights_l_crisp = eigenvector_method(crisp_matrix_l, Crisp)
    lower_weights = np.array([w.value for w in weights_l_crisp])

    # Step 3: Get the upper bound weights (alpha = 0.0)
    crisp_matrix_u = np.array([[interval[1] for interval in row] for row in alpha_0_cut])
    crisp_matrix_u = np.array([[Crisp(val) for val in row] for row in crisp_matrix_u], dtype=object)
    weights_u_crisp = eigenvector_method(crisp_matrix_u, Crisp)
    upper_weights = np.array([w.value for w in weights_u_crisp])

    # Step 4: Construct the final fuzzy weights
    fuzzy_weights = []
    for i in range(n):
        params = sorted([lower_weights[i], middle_weights[i], upper_weights[i]])
        if number_type.__name__ == 'TFN':
            fuzzy_weights.append(TFN(*params))
        elif number_type.__name__ == 'TrFN':
            # Convert a TFN-like result to a degenerate TrFN
            from .types import TrFN
            fuzzy_weights.append(TrFN(params[0], params[1], params[1], params[2]))
        # ... and so on for other types if they support this method

    return fuzzy_weights

@register_weight_method('TFN', 'fuzzy_programming')
def mikhailov_fuzzy_programming(matrix: np.ndarray, number_type: Type[TFN], **kwargs) -> Dict[str, Any]:
    """
    Derives crisp weights and a consistency index using the fuzzy programming
    method by Mikhailov (2004).

    .. note::
        **Academic Note:** This method transforms the weight derivation problem
        into a non-linear optimization problem. Its goal is to find the crisp
        weight vector that is "most consistent" with the original fuzzy judgments.
        The resulting lambda (λ) is a direct measure of consistency (λ > 0 is good).
    """
    _check_scipy_availability()
    n = matrix.shape[0]

    try:
        consistency_method = kwargs.get('consistency_method', 'centroid')
        initial_weights_results = derive_weights(matrix, number_type, method='geometric_mean', consistency_method=consistency_method)
        initial_weights = initial_weights_results['crisp_weights']
    except Exception:
        initial_weights = np.full(n, 1/n)

    # Initial guess for the optimizer: [w_0, ..., w_{n-1}, λ]
    initial_guess = np.append(initial_weights, 0.5) # Start with a reasonable lambda

    # Objective function: Minimize -lambda
    def objective(x):
        return -x[-1]

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - 1}]
    for i in range(n):
        for j in range(i + 1, n):
            tfn = matrix[i, j]
            # Ensure l, m, u are valid TFNs
            if not (tfn.l <= tfn.m <= tfn.u):
                raise ValueError(f"Invalid TFN at ({i},{j}): {tfn}. l <= m <= u must hold.")

            l, m, u = tfn.l, tfn.m, tfn.u
            # Constraint 1: (m-l)λw_j - w_i + l*w_j <= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j, l=l, m=m: -((m - l) * x[-1] * x[j] - x[i] + l * x[j])})
            # Constraint 2: (u-m)λw_j + w_i - u*w_j <= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j, u=u, m=m: -((u - m) * x[-1] * x[j] + x[i] - u * x[j])})

    # Bounds: w_i > 0 and λ >= 0
    bounds = [(1e-9, None)] * n + [(0, None)]

    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        return {
            "weights": [number_type.from_crisp(w) for w in initial_weights], # Return initial guess as a fallback
            "crisp_weights": initial_weights,
            "lambda_consistency": -1.0, # Failure signal
            "optimization_success": False,
            "optimization_message": f"Optimization failed: {result.message}. This often indicates high matrix inconsistency."
        }

    crisp_weights = result.x[:-1]
    lambda_consistency = result.x[-1]

    return {
        "weights": [number_type.from_crisp(w) for w in crisp_weights],
        "crisp_weights": crisp_weights / np.sum(crisp_weights), # Final normalization
        "lambda_consistency": lambda_consistency,
        "optimization_success": result.success,
        "optimization_message": result.message
    }


# ==============================================================================
# 4. THE PRIMARY DISPATCHER FUNCTION
# ==============================================================================

def derive_weights(
    matrix: np.ndarray,
    number_type: Type[Number],
    method: str = "geometric_mean",
    consistency_method: str = "centroid"
) -> Dict[str, Any]:
    """
    Derives weights from a comparison matrix using the specified method.
    This function acts as a dispatcher, selecting the appropriate algorithm
    based on the number type and chosen method.

    Args:
        matrix: The comparison matrix of shape (n, n).
        number_type: The class of the number type (e.g., Crisp, TFN, IFN).
        method: The weight derivation method to use.
            The method to use, one of:
            Crisp: "geometric_mean", "eigenvector"
            TFN: "geometric_mean", "extent_analysis", "llsm", "lambda_max", "fuzzy_programming"
            TrFN: "geometric_mean", "lambda_max", "llsm"
            GFN: "geometric_mean"
            IFN: "geometric_mean"
            IT2TrFN: "geometric_mean"
        consistency_method: Defuzzification method to use
    Returns:
        A list of derived weights.
    """
    type_name = number_type.__name__
    key = (type_name, method)

    derivation_func = WEIGHT_DERIVATION_REGISTRY.get(key)

    if derivation_func is None:
        available_methods = [m for (t, m) in WEIGHT_DERIVATION_REGISTRY.keys() if t == type_name]
        raise ValueError(
            f"Method '{method}' is not registered for number type '{type_name}'. Available methods for '{type_name}': {available_methods}"
        )

    result = derivation_func(matrix, number_type)

    if isinstance(result, dict):
        if 'weights' not in result or 'crisp_weights' not in result:
             raise TypeError(f"Registered method {derivation_func.__name__} must return a dict with 'weights' and 'crisp_weights'")
        return result
    elif isinstance(result, list):
        weights = result
        crisp_weights = np.array([w.defuzzify(method=consistency_method) for w in weights])
        return {
            "weights": weights,
            "crisp_weights": crisp_weights / np.sum(crisp_weights), # Normalize
            "possibility_matrix": None, # Not applicable
            "min_degrees": None # Not applicable
        }
    else:
        raise TypeError(f"Registered method {derivation_func.__name__} returned an unexpected type: {type(result)}")
