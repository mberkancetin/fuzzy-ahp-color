from __future__ import annotations
from typing import Dict, Callable, Any
import numpy as np

COMPLETION_METHOD_REGISTRY: Dict[str, Callable] = {}

def register_completion_method(name: str):
    """A decorator to register a new iPCM completion method."""
    def decorator(func: Callable) -> Callable:
        if name in COMPLETION_METHOD_REGISTRY:
            print(f"Warning: Overwriting completion method '{name}'")
        COMPLETION_METHOD_REGISTRY[name] = func
        return func
    return decorator

def complete_matrix(
    incomplete_matrix: np.ndarray,
    method: str = "eigenvalue_optimization",
    **kwargs
) -> np.ndarray:
    """
    Completes a pairwise comparison matrix with missing entries.

    Missing entries should be represented by `None` or `np.nan`.

    Args:
        incomplete_matrix: A square NumPy array of dtype=object containing
                           numbers and `None` for missing values.
        method: The name of the completion algorithm to use.
        **kwargs: Additional arguments to pass to the specific completion method.

    Returns:
        A completed, reciprocal NumPy array.
    """
    completion_func = COMPLETION_METHOD_REGISTRY.get(method)
    if not completion_func:
        raise ValueError(f"Completion method '{method}' not found. Available: {list(COMPLETION_METHOD_REGISTRY.keys())}")

    return completion_func(incomplete_matrix, **kwargs)

# ==============================================================================
# IMPLEMENTATION OF COMPLETION METHODS
# ==============================================================================

@register_completion_method("eigenvalue_optimization")
def complete_by_eigenvalue_optimization(
    incomplete_matrix: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-6,
    **kwargs
) -> np.ndarray:
    """
    Completes an iPCM by minimizing the principal eigenvalue (lambda_max).
    Based on the cyclic coordinates method (Bozóki et al., 2010).

    This is an iterative, optimization-based approach.
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        raise ImportError("The 'eigenvalue_optimization' method requires SciPy. Install it with: pip install scipy")

    n = incomplete_matrix.shape[0]
    matrix = incomplete_matrix.copy()
    missing_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] is None or np.isnan(matrix[i, j]):
                missing_indices.append((i, j))
                matrix[i, j] = 1.0
                matrix[j, i] = 1.0

    def get_lambda_max(x_val, i, j, current_matrix):
        temp_matrix = current_matrix.copy()
        temp_matrix[i, j] = x_val
        temp_matrix[j, i] = 1.0 / x_val
        numeric_matrix = temp_matrix.astype(float)
        return np.max(np.real(np.linalg.eigvals(numeric_matrix)))

    for iteration in range(max_iter):
        previous_matrix = matrix.copy().astype(float)

        for i, j in missing_indices:
            res = minimize_scalar(
                get_lambda_max,
                args=(i, j, matrix),
                bounds=(1e-5, 1e5),
                method='bounded'
            )
            matrix[i, j] = res.x
            matrix[j, i] = 1.0 / res.x

        if np.allclose(previous_matrix, matrix.astype(float), atol=tolerance):
            print(f"Eigenvalue optimization converged in {iteration + 1} iterations.")
            break
    else:
        print("Warning: Eigenvalue optimization did not converge within max_iter.")

    return matrix.astype(float)

@register_completion_method("dematel")
def complete_by_dematel(incomplete_matrix: np.ndarray, **kwargs) -> np.ndarray:
    """
    Completes an iPCM using the DEMATEL-based method (Zhou et al., 2018).
    This is a non-iterative, direct calculation method.
    """
    n = incomplete_matrix.shape[0]

    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if incomplete_matrix[i, j] is not None and not np.isnan(incomplete_matrix[i, j]):
                adj[i].append(j)
                adj[j].append(i)

    q = [0]
    visited = {0}
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                q.append(v)

    if len(visited) != n:
        raise ValueError("Matrix is disconnected. DEMATEL completion requires a connected graph of judgments.")

    # Convert iPCM to Direct Relation Matrix (DRM)
    drm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if incomplete_matrix[i, j] is not None and not np.isnan(incomplete_matrix[i, j]):
                drm[i, j] = incomplete_matrix[i, j]

    row_sums = np.sum(drm, axis=1)
    max_sum = np.max(row_sums)
    if max_sum == 0:
        return np.ones((n, n))

    normalized_drm = drm / max_sum

    identity = np.identity(n)
    try:
        trm = normalized_drm @ np.linalg.inv(identity - normalized_drm)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular, cannot compute TRM. The iPCM might be disconnected.")

    completed_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # The formula is
            # c_ij = sqrt(t_ij / t_ji)
            if trm[j, i] > 1e-9:
                ratio = trm[i, j] / trm[j, i]
                completed_matrix[i, j] = np.sqrt(ratio) if ratio > 0 else 1.0
                completed_matrix[j, i] = 1.0 / completed_matrix[i, j]
            else: 
                completed_matrix[i, j] = 1.0

    return completed_matrix.astype(float)

@register_completion_method("llsm")
def complete_by_llsm(incomplete_matrix: np.ndarray, **kwargs) -> np.ndarray:
    """
    Completes an iPCM using the Logarithmic Least Squares Method.

    This method finds the missing values that are most consistent with the known
    values in a geometric sense. It is a direct, non-iterative method based on
    solving a system of linear equations derived from the graph Laplacian.
    Source: Bozóki et al. (2010).
    """
    from scipy.linalg import pinv

    n = incomplete_matrix.shape[0]
    matrix_for_check = incomplete_matrix.copy()

    adj = {i: [] for i in range(n)}

    is_missing = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            cell = matrix_for_check[i, j]
            if cell is None or (isinstance(cell, float) and np.isnan(cell)):
                is_missing[i, j] = True

    for i in range(n):
        for j in range(i + 1, n):
            if matrix_for_check[i, j] is not None:
                adj[i].append(j)
                adj[j].append(i)

    q, visited, head = [0], {0}, 0
    while head < len(q):
        u = q[head]; head += 1
        for v in adj[u]:
            if v not in visited:
                visited.add(v); q.append(v)

    if len(visited) != n:
        raise ValueError("LLSM completion failed: iPCM graph may be disconnected.")

    laplacian = np.zeros((n, n))
    b = np.zeros(n)

    matrix_float = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if not is_missing[i, j]:
                matrix_float[i, j] = float(matrix_for_check[i, j])
            else:
                matrix_float[i, j] = np.nan

    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(matrix_float[i, j]):
                log_aij = np.log(matrix_float[i, j])

                laplacian[i, j] = -1
                laplacian[j, i] = -1
                laplacian[i, i] += 1
                laplacian[j, j] += 1

                b[i] += log_aij
                b[j] -= log_aij

    try:
        log_weights = pinv(laplacian) @ b
    except np.linalg.LinAlgError:
        raise ValueError("Could not solve LLSM system. The iPCM graph may be disconnected.")

    weights = np.exp(log_weights)

    completed_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix_float[i, j]):
                completed_matrix[i, j] = matrix_float[i, j]
            else:
                completed_matrix[i, j] = weights[i] / weights[j]

    return completed_matrix.astype(float)
