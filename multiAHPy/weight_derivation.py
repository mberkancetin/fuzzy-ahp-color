from __future__ import annotations
from typing import List, Type, Dict, Any, TYPE_CHECKING
import numpy as np
import math

if TYPE_CHECKING:
    from multiAHPy.types import NumericType, Number, TFN, Crisp, IFN, IT2TrFN

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
# 1. GENERIC & CLASSIC AHP ALGORITHMS
# ==============================================================================

def geometric_mean_method(matrix: np.ndarray, number_type: Type[Number], consistency_method: str) -> List[Number]:
    """
    Derives weights using the fuzzy geometric mean method.

    .. note::
        **Academic Note:** This method is a direct extension of the geometric mean
        used in classical AHP. It is generally considered a robust and stable method
        that correctly handles the ratio-scale nature of AHP judgments. It is often
        recommended for its mathematical consistency and reliable results.

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

def eigenvector_method(matrix: np.ndarray, number_type: Type[Crisp], max_iter=20, tol=1e-6) -> List[Number]:
    """
    Derives weights using the principal eigenvector method (Power Iteration).
    This implementation is for CRISP matrices only.

    Args:
        matrix: A crisp comparison matrix.
        number_type: Must be the Crisp class.
        max_iter: Maximum iterations for convergence.
        tol: Tolerance for convergence.

    Returns:
        A list of crisp weights.
    """
    if number_type.__name__ != 'Crisp':
        raise TypeError("Standard eigenvector method is only applicable to crisp matrices.")

    n = matrix.shape[0]
    # Extract float values from Crisp objects
    from multiAHPy.types import Crisp
    crisp_matrix = np.array([[Crisp(cell) for cell in row] for row in matrix])
    crisp_matrix = np.array([[cell.value for cell in row] for row in crisp_matrix])

    # Power method to find the principal eigenvector
    weights = np.ones(n)
    for _ in range(max_iter):
        weights_new = crisp_matrix @ weights
        weights_new /= np.linalg.norm(weights_new) # Normalize vector
        if np.allclose(weights, weights_new, atol=tol):
            break
        weights = weights_new

    # Final normalization so weights sum to 1
    normalized_weights = weights / np.sum(weights)

    # Return as a list of Crisp objects
    return [number_type(w) for w in normalized_weights]

# ==============================================================================
# 2. FUZZY-SPECIFIC AHP ALGORITHMS
# ==============================================================================

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

    Returns:
        A dictionary containing weights, crisp_weights, possibility_matrix, etc.
    """
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

def mikhailov_fuzzy_programming(matrix: np.ndarray, number_type: Type[TFN]) -> Dict[str, Any]:
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

    # The variables for the optimizer will be [w_0, w_1, ..., w_{n-1}, λ]
    # So there are n+1 variables in total.

    # Objective function: We want to MAXIMIZE lambda, which is equivalent
    # to MINIMIZING -lambda. Our variable vector x will have lambda as its last element.
    def objective(x):
        return -x[-1] # Minimize -λ

    # Constraints
    constraints = []
    # 1. Sum of weights must be 1: Σw_i - 1 = 0
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - 1})

    # 2. Add the two main constraints for each upper-triangle fuzzy judgment
    for i in range(n):
        for j in range(i + 1, n):
            tfn = matrix[i, j]
            l, m, u = tfn.l, tfn.m, tfn.u

            # Constraint 1: (m-l)λw_j - w_i + l*w_j <= 0
            cons1 = {'type': 'ineq', 'fun': lambda x, i=i, j=j, l=l, m=m: -((m-l)*x[-1]*x[j] - x[i] + l*x[j])}

            # Constraint 2: (u-m)λw_j + w_i - u*w_j <= 0
            cons2 = {'type': 'ineq', 'fun': lambda x, i=i, j=j, u=u, m=m: -((u-m)*x[-1]*x[j] + x[i] - u*x[j])}

            constraints.append(cons1)
            constraints.append(cons2)

    # Bounds for the variables: w_i > 0 and λ >= 0
    bounds = [(1e-9, None) for _ in range(n)] + [(0, None)] # w_i are strictly positive, λ can be 0

    # Initial guess for the variables
    initial_guess = np.array([1/n] * n + [0.5]) # Equal weights, lambda=0.5

    # Run the optimizer
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        print(f"Warning: Fuzzy programming optimization failed or did not converge. Reason: {result.message}")

    # Extract the results
    crisp_weights = result.x[:-1]
    lambda_consistency = result.x[-1]

    return {
        "weights": [number_type.from_crisp(w) for w in crisp_weights],
        "crisp_weights": crisp_weights,
        "lambda_consistency": lambda_consistency,
        "optimization_success": result.success,
        "optimization_message": result.message
    }


# ==============================================================================
# 3. THE PRIMARY DISPATCHER FUNCTION
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

    weights = []
    # --- Route to classic AHP methods for Crisp type ---
    if type_name == 'Crisp':
        from .types import TFN, TrFN, Crisp, GFN, NumericType, Number, IFN, IT2TrFN
        if method == 'geometric_mean':
            weights =  geometric_mean_method(matrix, number_type, consistency_method=consistency_method)
        elif method == 'eigenvector':
            weights =  eigenvector_method(matrix, number_type)
        else:
            raise ValueError(f"Method '{method}' is not supported for Crisp matrices. Use 'geometric_mean' or 'eigenvector'.")

    # --- Route to fuzzy methods for TFN type ---
    elif type_name == 'TFN':
        from .types import TFN, TrFN, Crisp, GFN, NumericType, Number
        if method == 'geometric_mean':
            weights =  geometric_mean_method(matrix, number_type, consistency_method=consistency_method)
        elif method == 'extent_analysis':
            if not isinstance(matrix[0,0], TFN):
                 raise TypeError("Cannot use 'extent_analysis' on non-TFN matrix.")
            return extent_analysis_method(matrix, number_type)
        elif method == 'llsm':
            weights =  fuzzy_llsm_method(matrix, number_type, components=['l', 'm', 'u'])
        elif method == 'lambda_max':
            weights = lambda_max_method(matrix, number_type)
        elif method == 'fuzzy_programming':
            return mikhailov_fuzzy_programming(matrix, number_type)
        else:
            raise ValueError(f"Method '{method}' is not supported for TFN. Use 'geometric_mean', 'extent_analysis', 'llsm', 'lambda_max', or 'fuzzy_programming'.")

    # --- Route to fuzzy methods for TrFN type ---
    elif type_name == 'TrFN':
        from .types import TFN, TrFN, Crisp, GFN, NumericType, Number
        if method == 'geometric_mean':
            weights =  geometric_mean_method(matrix, number_type, consistency_method=consistency_method)
        elif method == 'llsm':
            weights =  fuzzy_llsm_method(matrix, number_type, components=['a', 'b', 'c', 'd'])
        elif method == 'lambda_max':
            weights = lambda_max_method(matrix, number_type)
        else:
            raise ValueError(f"Method '{method}' is not supported for TrFN. Use 'geometric_mean', 'llsm' or 'lambda_max'.")

    # --- Route to fuzzy methods for GFN type ---
    elif type_name == 'GFN':
        from .types import TFN, TrFN, Crisp, GFN, NumericType, Number
        if method == 'geometric_mean':
            weights =  geometric_mean_method(matrix, number_type, consistency_method=consistency_method)
        else:
            raise ValueError(f"Method '{method}' is not supported for GFN. Currently only 'geometric_mean' is available.")

    # --- Route to fuzzy methods for IT2TrFN type ---
    elif type_name == 'IT2TrFN':
        from .types import IT2TrFN
        if method == 'geometric_mean':
            weights =  geometric_mean_method(matrix, number_type, consistency_method=consistency_method)
        else:
            raise ValueError(f"Method '{method}' is not supported for IT2TrFN. Currently only 'geometric_mean' is available.")

    # --- Route to fuzzy methods for IFN type ---
    elif type_name == 'IFN':
        from .types import IFN
        if method == 'geometric_mean':
            """
            .. note::
                **Academic Note:** This applies the geometric mean method directly
                to an aggregated IFN matrix. This is one valid approach. Another
                common approach in the literature involves calculating crisp weights
                for each expert first, then aggregating those priorities using
                a method like IFWA (see the 'aggregation' module).
            """
            weights =  geometric_mean_method(matrix, number_type, consistency_method=consistency_method)
        else:
            raise ValueError(f"Method '{method}' is not supported for IFN. Currently only 'geometric_mean' is available.")

    else:
        raise TypeError(f"Weight derivation not implemented for number type: {type_name}")

    crisp_weights = np.array([w.defuzzify(method=consistency_method) for w in weights])
    return {
        "weights": weights,
        "crisp_weights": crisp_weights / np.sum(crisp_weights), # Normalize
        "possibility_matrix": None, # Not applicable
        "min_degrees": None # Not applicable
    }
