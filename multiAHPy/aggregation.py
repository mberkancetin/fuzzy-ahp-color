from __future__ import annotations
from typing import List, Type, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from multiAHPy.types import NumericType, Number, TFN, TrFN
    from multiAHPy.matrix_builder import create_comparison_matrix

def aggregate_matrices(
    matrices: List[np.ndarray],
    method: str = "geometric",
    expert_weights: List[float] | None = None
) -> np.ndarray:
    """
    Aggregates a list of expert judgment matrices into a single group matrix.

    This function combines multiple comparison matrices from different participants
    into a single representative matrix using a specified aggregation technique.

    Args:
        matrices: A list of comparison matrices. Each matrix should be a NumPy
                  array of NumericType objects (e.g., Crisp, TFN, TrFN).
        method (str, optional): The aggregation method to use.
                                Options: "arithmetic", "geometric", "median", "min_max".
                                Defaults to "geometric", which is generally preferred.
        expert_weights (List[float], optional): A list of weights corresponding to each
                                                 expert/matrix. If None, equal weights
                                                 are assumed. Not used for 'median'
                                                 or 'min_max' methods.

    Returns:
        A single aggregated comparison matrix as a NumPy array.

    Raises:
        ValueError: If the list of matrices is empty, matrices have different shapes,
                    expert weights are invalid, or an unknown method is specified.
        TypeError: If a method like 'median' is used on an unsupported number type.
    """
    if not matrices:
        raise ValueError("The list of matrices to aggregate cannot be empty.")

    num_matrices = len(matrices)
    first_matrix = matrices[0]
    n = first_matrix.shape[0]
    number_type = type(first_matrix[0, 0])

    # --- Initial Validation ---
    for matrix in matrices[1:]:
        if matrix.shape != first_matrix.shape:
            raise ValueError("All matrices must have the same dimensions for aggregation.")

    # Validate and normalize expert weights
    if expert_weights is None:
        weights = [1.0 / num_matrices] * num_matrices
    else:
        if len(expert_weights) != num_matrices:
            raise ValueError("Number of expert weights must match the number of matrices.")
        weight_sum = sum(expert_weights)
        if abs(weight_sum) < 1e-9:
            raise ValueError("Sum of expert weights cannot be zero.")
        weights = [w / weight_sum for w in expert_weights]

    # --- Dispatch to the correct aggregation method ---

    # Initialize the aggregated matrix using a helper from the matrix_factory module
    from .matrix_builder import create_comparison_matrix
    aggregated_matrix = create_comparison_matrix(n, number_type)

    for i in range(n):
        for j in range(n):
            if i == j: continue

            if method == "geometric":
                # Weighted geometric mean: product of (matrix_k ^ weight_k)
                agg_cell = number_type.multiplicative_identity()
                for k, matrix in enumerate(matrices):
                    agg_cell *= matrix[i, j] ** weights[k]
                aggregated_matrix[i, j] = agg_cell

            elif method == "arithmetic":
                # Weighted arithmetic mean: sum of (matrix_k * weight_k)
                agg_cell = number_type.neutral_element()
                for k, matrix in enumerate(matrices):
                    agg_cell += matrix[i, j] * weights[k]
                aggregated_matrix[i, j] = agg_cell

            elif method == "median":
                if not hasattr(matrices[0][0,0], '__dict__'):
                    raise TypeError("Median aggregation requires component-wise fuzzy numbers (e.g., TFN, TrFN).")

                components = list(matrices[0][0,0].__dict__.keys())
                median_params = [
                    np.median([getattr(matrix[i, j], comp_name) for matrix in matrices])
                    for comp_name in components
                ]
                aggregated_matrix[i, j] = number_type(*median_params)

            elif method == "min_max":
                if not hasattr(matrices[0][0,0], '__dict__') or len(matrices[0][0,0].__dict__) < 3:
                    raise TypeError("Min-Max aggregation requires fuzzy numbers with at least 3 components (e.g., TFN, TrFN).")

                components = list(matrices[0][0,0].__dict__.keys())
                l_values = [getattr(m[i, j], components[0]) for m in matrices]
                u_values = [getattr(m[i, j], components[-1]) for m in matrices]

                # Find min of lower, mean of middle(s), max of upper
                agg_l, agg_u = min(l_values), max(u_values)

                if len(components) == 4: # TrFN case
                    m_values = [getattr(m[i, j], components[1]) for m in matrices]
                    c_values = [getattr(m[i, j], components[2]) for m in matrices]
                    agg_m, agg_c = np.mean(m_values), np.mean(c_values)
                    aggregated_matrix[i, j] = number_type(agg_l, agg_m, agg_c, agg_u)
                else: # TFN case
                    m_values = [getattr(m[i, j], components[1]) for m in matrices]
                    agg_m = np.mean(m_values)
                    aggregated_matrix[i, j] = number_type(agg_l, agg_m, agg_u)

            else:
                raise ValueError(f"Unknown aggregation method: '{method}'. "
                                 "Available: 'arithmetic', 'geometric', 'median', 'min_max'.")

    return aggregated_matrix
