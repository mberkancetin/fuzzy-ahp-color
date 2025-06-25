from __future__ import annotations
from typing import List, Type, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .types import NumericType, Number, TFN, TrFN, IFN, IT2TrFN
    from .matrix_builder import create_comparison_matrix

# ==============================================================================
# AGGREGATION OF JUDGMENTS
# ==============================================================================

def _ifn_similarity(ifn1: IFN, ifn2: IFN) -> float:
    """Calculates a similarity score between two IFNs based on distance."""
    # Using normalized Hamming distance
    distance = 0.5 * (abs(ifn1.mu - ifn2.mu) + abs(ifn1.nu - ifn2.nu) + abs(ifn1.pi - ifn2.pi))
    return 1 - distance

def aggregate_matrices(
    matrices: List[np.ndarray],
    method: str = "geometric",
    expert_weights: List[float] | None = None
) -> np.ndarray:
    """
    Aggregates a list of expert judgment matrices into a single group matrix.

    This function combines multiple comparison matrices from different participants
    into a single representative matrix using a specified aggregation technique.

    .. note::
        **Academic Note on Aggregation Methods:**
        - **`geometric` (Default & Recommended):** The geometric mean is generally
          preferred for aggregating judgments in AHP as it preserves the reciprocal
          property of the matrices well and is less sensitive to extreme values.
        - **`arithmetic`:** The arithmetic mean is simpler to understand but can be
          unduly influenced by outlier judgments.
        - **`median` & `min_max`:** These are robust statistical methods that can be useful
          for understanding the range and central tendency of expert disagreement,
          but they do not have the same theoretical foundation for AHP as the
          geometric mean.

    Args:
        matrices: A list of comparison matrices. Each matrix should be a NumPy
                  array of NumericType objects (e.g., Crisp, TFN, TrFN).
        method (str, optional): The aggregation method to use.
                                For TFN/TrFN/Crisp: 'geometric', 'arithmetic', 'median', 'min_max'.
                                For IFN: 'ifwa' (Intuitionistic Fuzzy Weighted Average), 'consensus'.
                                Defaults to "geometric", which is generally preferred.
        expert_weights: Weights for each expert. Used in 'ifwa' and weighted means. If None, equal weights
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

    if number_type.__name__ == 'IFN':
            if method == 'ifwa':
                return _aggregate_ifn_ifwa(matrices, n, number_type, expert_weights)
            elif method == 'consensus':
                return _aggregate_ifn_consensus(matrices, n, number_type)
            else:
                raise ValueError(f"Unsupported method '{method}' for IFN. Use 'ifwa' or 'consensus'.")
    else:
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
                    # --- Rationale Comment ---
                    # This method creates an "envelope" of all expert judgments.
                    # The lower and upper bounds are the absolute min/max across all experts.
                    # For the middle component(s), we use the arithmetic mean as a measure
                    # of central tendency, a common approach cited in literature
                    # (e.g., Awasthi et al., 2018; Prakash & Barua, 2016a).

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

def _aggregate_ifn_ifwa(matrices: List[np.ndarray], n: int, number_type: Type[IFN], weights: List[float]) -> np.ndarray:
    """Aggregates IFN matrices using the Intuitionistic Fuzzy Weighted Average."""
    aggregated_matrix = create_comparison_matrix(n, number_type)
    for i in range(n):
        for j in range(n):
            if i == j: continue

            # Formula:
            # (1 - PRODUCT((1-mu_k)^w_k), PRODUCT(nu_k^w_k))
            # PRODUCT((1-mu_k)^w_k)
            prod_1_minus_mu = np.prod([(1 - m[i,j].mu) ** w for m, w in zip(matrices, weights)])

            # PRODUCT(nu_k^w_k)
            prod_nu = np.prod([m[i,j].nu ** w for m, w in zip(matrices, weights)])

            agg_mu = 1 - prod_1_minus_mu
            agg_nu = prod_nu

            aggregated_matrix[i, j] = number_type(agg_mu, agg_nu)

    return aggregated_matrix

def _aggregate_ifn_consensus(matrices: List[np.ndarray], n: int, number_type: Type[IFN]) -> np.ndarray:
    """
    Aggregates IFN matrices based on the consensus degree between experts.
    This method does not use pre-assigned expert weights.
    """
    num_experts = len(matrices)
    if num_experts < 2:
        return matrices[0] # No consensus to calculate with one expert

    # Step 1: Calculate the similarity matrix between all pairs of experts
    # For simplicity, we calculate an overall similarity score for each expert pair
    expert_similarity_matrix = np.ones((num_experts, num_experts))
    for k1 in range(num_experts):
        for k2 in range(k1 + 1, num_experts):
            # Average the similarity across all cells in the two matrices
            similarities = [_ifn_similarity(matrices[k1][i,j], matrices[k2][i,j])
                            for i in range(n) for j in range(n)]
            avg_sim = np.mean(similarities)
            expert_similarity_matrix[k1, k2] = expert_similarity_matrix[k2, k1] = avg_sim

    # Step 2: Calculate the average agreement (support) for each expert
    agreement_scores = np.sum(expert_similarity_matrix, axis=1) / (num_experts - 1)

    # Step 3: Calculate the consensus degree coefficient (CDC) for each expert
    # This is just the normalized agreement score.
    total_agreement = np.sum(agreement_scores)
    consensus_weights = agreement_scores / total_agreement if total_agreement > 0 else [1/num_experts]*num_experts

    print(f"  - Consensus weights calculated for experts: {[f'{w:.3f}' for w in consensus_weights]}")

    # Step 4: Aggregate using the consensus weights (using the IFWA method)
    return _aggregate_ifn_ifwa(matrices, n, number_type, consensus_weights)

def aggregate_priorities_ifwa(
    priorities: List[IFN],
    expert_weights: List[float] | None = None
) -> IFN:
    """
    Aggregates a list of Intuitionistic Fuzzy Numbers (IFNs) using the
    Intuitionistic Fuzzy Weighted Averaging (IFWA) operator.

    This is typically used in group decision-making where each IFN represents
    the priority of a single item as determined by different experts, or it
    can be used to aggregate performance scores for an alternative under
    one criterion.

    Args:
        priorities: A list of IFN objects to be aggregated.
        expert_weights (List[float], optional): A list of weights for each IFN/expert.
                                                 If None, equal weights are assumed.

    Returns:
        A single, aggregated IFN representing the group priority.
    """
    num_priorities = len(priorities)
    if not priorities:
        raise ValueError("Priority list cannot be empty for aggregation.")

    if expert_weights is None:
        # If no weights are provided, assume all experts are equal.
        weights = [1.0 / num_priorities] * num_priorities
    else:
        # If weights are provided, validate them.
        if len(expert_weights) != num_priorities:
            raise ValueError(f"The number of expert weights provided ({len(expert_weights)}) "
                             f"must match the number of priorities ({num_priorities}).")

        weight_sum = sum(expert_weights)
        if abs(weight_sum) < 1e-9:
             raise ValueError("The sum of expert weights cannot be zero.")

        # Normalize weights so they sum to 1.
        weights = [w / weight_sum for w in expert_weights]

    # Formula for IFWA:
    # (1 - PRODUCT((1-mu_k)^w_k), PRODUCT(nu_k^w_k))
    prod_1_minus_mu = np.prod([(1 - p.mu) ** w for p, w in zip(priorities, weights)])
    prod_nu = np.prod([p.nu ** w for p, w in zip(priorities, weights)])

    agg_mu = 1 - prod_1_minus_mu
    agg_nu = prod_nu

    # Import locally to avoid potential circular dependencies if this file is imported elsewhere
    from .types import IFN
    return IFN(agg_mu, agg_nu)


# ==============================================================================
# AGGREGATION OF PRIORITIES
# ==============================================================================

def aggregate_priorities(
    matrices: List[np.ndarray],
    derivation_method: str = "geometric_mean",
    expert_weights: List[float] | None = None
) -> np.ndarray:
    """
    Aggregates priorities by first calculating individual weight vectors for each
    expert's matrix and then combining these weight vectors.
    This corresponds to the "Aggregation of Priorities" workflow.

    .. note::
        This method is useful when you want to weigh the final calculated
        priorities of experts, rather than their initial judgments.

    Args:
        matrices: A list of comparison matrices from participants.
        derivation_method (str): The method used to derive weights for each individual.
        expert_weights (List[float], optional): A list of weights for each expert's
                                                 final priority vector. If None,
                                                 equal weights are assumed.

    Returns:
        A single, aggregated crisp priority vector (NumPy array).
    """
    if not matrices:
        raise ValueError("Matrix list cannot be empty.")

    # --- Step 1: Calculate Individual Priority Vectors ---
    from .weight_derivation import derive_weights # Local import to avoid circular dependency

    individual_crisp_weights = []
    for matrix in matrices:
        number_type = type(matrix[0, 0])
        # Derive weights for the current matrix
        results = derive_weights(matrix, number_type, method=derivation_method)
        # We need the crisp weights for aggregation
        individual_crisp_weights.append(results['crisp_weights'])

    # We now have a list of weight vectors, e.g., [[0.6, 0.4], [0.7, 0.3]]
    # Convert to a 2D NumPy array for easier computation: (num_experts x num_criteria)
    weights_matrix = np.array(individual_crisp_weights)

    # --- Step 2: Aggregate the Priority Vectors ---
    # Validate and normalize expert weights
    num_experts = len(matrices)
    if expert_weights is None:
        weights = np.full(num_experts, 1.0 / num_experts)
    else:
        if len(expert_weights) != num_experts:
            raise ValueError("Number of expert weights must match the number of matrices.")
        weight_sum = sum(expert_weights)
        if abs(weight_sum) < 1e-9:
            raise ValueError("Sum of expert weights cannot be zero.")
        weights = np.array(expert_weights) / weight_sum

    # --- Step 3: Calculate the Weighted Average ---
    # Use np.average with the 'weights' parameter for a clean, weighted average.
    # axis=0 calculates the average down each column (for each criterion).
    final_group_priorities = np.average(weights_matrix, axis=0, weights=weights)

    # Final normalization to ensure it sums perfectly to 1
    return final_group_priorities / np.sum(final_group_priorities)
