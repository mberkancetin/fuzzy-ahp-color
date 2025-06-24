from __future__ import annotations
from typing import List, Type, TYPE_CHECKING
import numpy as np
import math

if TYPE_CHECKING:
    from multiAHPy.types import NumericType, Number, TFN, Crisp

# ==============================================================================
# 1. GENERIC & CLASSIC AHP ALGORITHMS
# ==============================================================================

def geometric_mean_method(matrix: np.ndarray, number_type: Type[Number]) -> List[Number]:
    """
    Derives weights using the geometric mean method. Works for both crisp and fuzzy numbers.

    Args:
        matrix: The comparison matrix of shape (n, n).
        number_type: The class of the number type being used (e.g., Crisp, TFN).

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
    if abs(total_sum.defuzzify()) < 1e-9:
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
    Chang's extent analysis method. Returns a dictionary with all intermediate results.
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


# ==============================================================================
# 3. THE PRIMARY DISPATCHER FUNCTION
# ==============================================================================

def derive_weights(matrix: np.ndarray, number_type: Type[Number], method: str = "geometric_mean") -> List[Number]:
    """
    Derives weights from a comparison matrix using the specified method.
    This function acts as a dispatcher, selecting the appropriate algorithm
    based on the number type and chosen method.

    Args:
        matrix: The comparison matrix of shape (n, n).
        number_type: The class of the number type (e.g., Crisp, TFN).
        method: The weight derivation method to use.
            The method to use, one of:
            Crisp: "geometric_mean", "eigenvector"
            TFN: "extent_analysis", "geometric_mean", "llsm"
            TrFN: "extent_analysis", "geometric_mean", "llsm"
            GFN: "geometric_mean"
    Returns:
        A list of derived weights.
    """
    type_name = number_type.__name__

    # --- Route to classic AHP methods for Crisp type ---
    if type_name == 'Crisp':
        from .types import TFN, TrFN, Crisp, GFN, NumericType, Number
        if method == 'geometric_mean':
            return geometric_mean_method(matrix, number_type)
        elif method == 'eigenvector':
            return eigenvector_method(matrix, number_type)
        else:
            raise ValueError(f"Method '{method}' is not supported for Crisp matrices. Use 'geometric_mean' or 'eigenvector'.")

    # --- Route to fuzzy methods for TFN type ---
    elif type_name == 'TFN':
        from .types import TFN, TrFN, Crisp, GFN, NumericType, Number
        if method == 'geometric_mean':
            return geometric_mean_method(matrix, number_type)
        elif method == 'extent_analysis':
            if not isinstance(matrix[0,0], TFN):
                 raise TypeError("Cannot use 'extent_analysis' on non-TFN matrix.")
            return extent_analysis_method(matrix, number_type)
        elif method == 'llsm':
            return fuzzy_llsm_method(matrix, number_type, components=['l', 'm', 'u'])
        else:
            raise ValueError(f"Method '{method}' is not supported for TFN. Use 'geometric_mean', 'extent_analysis', or 'llsm'.")

    # --- Route to fuzzy methods for TrFN type ---
    elif type_name == 'TrFN':
        from .types import TFN, TrFN, Crisp, GFN, NumericType, Number
        if method == 'geometric_mean':
            return geometric_mean_method(matrix, number_type)
        elif method == 'llsm':
            return fuzzy_llsm_method(matrix, number_type, components=['a', 'b', 'c', 'd'])
        else:
            raise ValueError(f"Method '{method}' is not supported for TrFN. Use 'geometric_mean' or 'llsm'.")

    # --- Route to fuzzy methods for GFN type ---
    elif type_name == 'GFN':
        from .types import TFN, TrFN, Crisp, GFN, NumericType, Number
        if method == 'geometric_mean':
            return geometric_mean_method(matrix, number_type)
        else:
            raise ValueError(f"Method '{method}' is not supported for GFN. Currently only 'geometric_mean' is available.")

    else:
        raise TypeError(f"Weight derivation not implemented for number type: {type_name}")
