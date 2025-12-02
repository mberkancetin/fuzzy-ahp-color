from __future__ import annotations
from typing import List, Type, TYPE_CHECKING, Callable
import numpy as np
from .config import configure_parameters

if TYPE_CHECKING:
    from .types import NumericType, Number, TFN, TrFN, IFN, IT2TrFN
    from .matrix_builder import create_comparison_matrix


AGGREGATION_REGISTRY = {}

def register_aggregation_method(number_type_name: str, method_name: str):
    """A decorator to register a new matrix aggregation method."""
    def decorator(func: Callable) -> Callable:
        if (number_type_name, method_name) in AGGREGATION_REGISTRY:
            print(f"Warning: Overwriting aggregation method '{method_name}'")
        AGGREGATION_REGISTRY[(number_type_name, method_name)] = func
        return func
    return decorator

def _ifn_similarity(ifn1: IFN, ifn2: IFN) -> float:
    """Calculates a similarity score between two IFNs based on distance."""
    # Using normalized Hamming distance
    distance = 0.5 * (abs(ifn1.mu - ifn2.mu) + abs(ifn1.nu - ifn2.nu) + abs(ifn1.pi - ifn2.pi))
    return 1 - distance


# ==============================================================================
# AGGREGATION OF JUDGMENTS
# ==============================================================================

def aggregate_matrices(
    matrices: List[np.ndarray],
    method: str = "geometric",
    expert_weights: List[float] | None = None,
    number_type: Type[NumericType] | None = None,
    tolerance: float | None = None
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

    if number_type is None:
        first_matrix = matrices[0]
        number_type_to_use = type(first_matrix[0, 0])
        print(f"Warning: number_type not provided to aggregate_matrices. Inferring type as {number_type_to_use.__name__}.")
    else:
        number_type_to_use = number_type

    final_tolerance = tolerance if tolerance is not None else configure_parameters.FLOAT_TOLERANCE

    num_matrices = len(matrices)
    first_matrix = matrices[0]
    n = first_matrix.shape[0]

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
        if abs(weight_sum) < final_tolerance:
            raise ValueError("Sum of expert weights cannot be zero.")
        weights = [w / weight_sum for w in expert_weights]

    # --- Dispatch using the Registry ---
    type_name = number_type_to_use.__name__
    key = (type_name, method)

    aggregation_func = AGGREGATION_REGISTRY.get(key)

    if aggregation_func is None:
        raise ValueError(
            f"Unknown aggregation method: '{method}' for '{type_name}'. Available methods: {list(AGGREGATION_REGISTRY.keys())}"
        )

    return aggregation_func(matrices=matrices, n=n, number_type=number_type_to_use, weights=weights)


@register_aggregation_method('TFN', 'geometric')
@register_aggregation_method('TrFN', 'geometric')
@register_aggregation_method('GFN', 'geometric')
@register_aggregation_method('IT2TrFN', 'geometric')
@register_aggregation_method('IFN', 'geometric')
@register_aggregation_method('Crisp', 'geometric')
def _aggregate_geometric(matrices: List[np.ndarray], n: int, number_type: Type[IFN], weights: List[float]) -> np.ndarray:
    from .matrix_builder import create_comparison_matrix
    aggregated_matrix = create_comparison_matrix(n, number_type)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            agg_cell = number_type.multiplicative_identity()
            for k, matrix in enumerate(matrices):
                agg_cell *= matrix[i, j] ** weights[k]
            aggregated_matrix[i, j] = agg_cell
    return aggregated_matrix

@register_aggregation_method('TFN', 'arithmetic')
@register_aggregation_method('TrFN', 'arithmetic')
@register_aggregation_method('Crisp', 'arithmetic')
def _aggregate_arithmetic(matrices: List[np.ndarray], n: int, number_type: Type[IFN], weights: List[float]) -> np.ndarray:
    from .matrix_builder import create_comparison_matrix
    aggregated_matrix = create_comparison_matrix(n, number_type)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            agg_cell = number_type.neutral_element()
            for k, matrix in enumerate(matrices):
                agg_cell += matrix[i, j] * weights[k]
            aggregated_matrix[i, j] = agg_cell
    return aggregated_matrix

@register_aggregation_method('TFN', 'median')
@register_aggregation_method('TrFN', 'median')
@register_aggregation_method('Crisp', 'median')
def _aggregate_median(matrices: List[np.ndarray], n: int, number_type: Type[IFN], weights: List[float]) -> np.ndarray:
    from .matrix_builder import create_comparison_matrix
    aggregated_matrix = create_comparison_matrix(n, number_type)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if not hasattr(matrices[0][0,0], '__dict__'):
                raise TypeError("Median aggregation requires component-wise fuzzy numbers (e.g., TFN, TrFN).")

            components = list(matrices[0][0,0].__dict__.keys())
            median_params = [
                np.median([getattr(matrix[i, j], comp_name) for matrix in matrices])
                for comp_name in components
            ]
            aggregated_matrix[i, j] = number_type(*median_params)
    return aggregated_matrix

@register_aggregation_method('TFN', 'min_max')
@register_aggregation_method('TrFN', 'min_max')
@register_aggregation_method('Crisp', 'min_max')
def _aggregate_min_max(matrices: List[np.ndarray], n: int, number_type: Type[IFN], weights: List[float]) -> np.ndarray:
    from .matrix_builder import create_comparison_matrix
    aggregated_matrix = create_comparison_matrix(n, number_type)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if not hasattr(matrices[0][0,0], '__dict__') or len(matrices[0][0,0].__dict__) < 3:
                raise TypeError("Min-Max aggregation requires fuzzy numbers with at least 3 components (e.g., TFN, TrFN).")

            components = list(matrices[0][0,0].__dict__.keys())
            l_values = [getattr(m[i, j], components[0]) for m in matrices]
            u_values = [getattr(m[i, j], components[-1]) for m in matrices]

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
    return aggregated_matrix

@register_aggregation_method('IFN', 'ifwa')
def _aggregate_ifn_ifwa(matrices: List[np.ndarray], n: int, number_type: Type[IFN], weights: List[float]) -> np.ndarray:
    """Aggregates IFN matrices using the Intuitionistic Fuzzy Weighted Average."""
    from .matrix_builder import create_comparison_matrix
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

@register_aggregation_method('IFN', 'consensus')
def _aggregate_ifn_consensus(matrices: List[np.ndarray], n: int, number_type: Type[IFN], weights: List[float]) -> np.ndarray:
    """
    Aggregates IFN matrices based on the consensus degree between experts.
    This method does not use pre-assigned expert weights.
    """
    num_experts = len(matrices)
    if num_experts < 2:
        return matrices[0] # No consensus to calculate with one expert

    # Step 1: Calculate the similarity matrix between all pairs of experts
    expert_similarity_matrix = np.ones((num_experts, num_experts))
    for k1 in range(num_experts):
        for k2 in range(k1 + 1, num_experts):
            similarities = [_ifn_similarity(matrices[k1][i,j], matrices[k2][i,j])
                            for i in range(n) for j in range(n)]
            avg_sim = np.mean(similarities)
            expert_similarity_matrix[k1, k2] = expert_similarity_matrix[k2, k1] = avg_sim

    # Step 2: Calculate the average agreement (support) for each expert
    agreement_scores = np.sum(expert_similarity_matrix, axis=1) / (num_experts - 1)

    # Step 3: Calculate the consensus degree coefficient (CDC) for each expert
    total_agreement = np.sum(agreement_scores)
    consensus_weights = agreement_scores / total_agreement if total_agreement > 0 else [1/num_experts]*num_experts

    print(f"  - Consensus weights calculated for experts: {[f'{w:.3f}' for w in consensus_weights]}")

    # Step 4: Aggregate using the consensus weights (using the IFWA method)
    return _aggregate_ifn_ifwa(matrices, n, number_type, consensus_weights)

@register_aggregation_method('IFN', 'ifowa')
def _aggregate_priorities_ifowa_operator(
    priorities: List[IFN],
    ordered_weights: List[float] | None = None,
    tolerance: float | None = None,
    **kwargs
) -> IFN:
    """
    Aggregates a list of IFNs using the Intuitionistic Fuzzy Ordered
    Weighted Averaging (IFOWA) operator, from Xu (2007).

    This operator re-orders the IFNs from largest to smallest before applying
    the weights. The weights correspond to the rank/position, not the source.

    Args:
        priorities: A list of IFN objects to be aggregated.
        ordered_weights: A list of weights for the ordered positions. Must sum to 1.
                         If None, equal weights are assumed.
    """
    num_priorities = len(priorities)
    if not priorities:
        raise ValueError("Priority list cannot be empty.")

    final_tolerance = tolerance if tolerance is not None else configure_parameters.FLOAT_TOLERANCE

    sorted_priorities = sorted(priorities, reverse=True)

    if ordered_weights is None:
        weights = [1.0 / num_priorities] * num_priorities
    else:
        if len(ordered_weights) != num_priorities:
            raise ValueError(f"Number of ordered weights ({len(ordered_weights)}) must match number of priorities ({num_priorities}).")
        weight_sum = sum(ordered_weights)
        if abs(weight_sum) < final_tolerance:
             raise ValueError("Sum of ordered weights cannot be zero.")
        weights = [w / weight_sum for w in ordered_weights]

    # Apply the IFWA formula to the *sorted* priorities
    prod_1_minus_mu = np.prod([(1 - p.mu) ** w for p, w in zip(sorted_priorities, weights)])
    prod_nu = np.prod([p.nu ** w for p, w in zip(sorted_priorities, weights)])

    agg_mu = 1 - prod_1_minus_mu
    agg_nu = prod_nu

    return IFN(agg_mu, agg_nu)

@register_aggregation_method('IFN', 'ifha')
def _aggregate_priorities_ifha_operator(
    priorities: List[IFN],
    expert_weights: List[float],
    ordered_weights: List[float],
    n_balance: int | None = None,
    **kwargs
) -> IFN:
    """
    Aggregates IFNs using the Intuitionistic Fuzzy Hybrid Aggregation
    (IFHA) operator, from Xu (2007). This is a two-layer operator.

    Args:
        priorities: A list of IFN objects from different experts.
        expert_weights: Weights corresponding to each expert/source.
        ordered_weights: Weights corresponding to the ordered positions.
        n_balance: The balancing coefficient (typically the number of priorities).
    """
    num_priorities = len(priorities)
    if n_balance is None:
        n_balance = num_priorities

    # Step 1: Calculate the weighted IFNs (å = n * w * ã)
    # The paper's formula is
    # å_j = n_balance * w_j * ã_j
    weighted_priorities = []
    for i in range(num_priorities):
        scalar = n_balance * expert_weights[i]
        p = priorities[i]
        weighted_p = p.scale(scalar)
        weighted_priorities.append(weighted_p)

    # Step 2: Sort the *weighted* priorities in descending order
    sorted_weighted_priorities = sorted(weighted_priorities, reverse=True)

    # Step 3: Apply the IFOWA logic to the sorted, weighted priorities using the ordered_weights
    return _aggregate_priorities_ifowa_operator(sorted_weighted_priorities, ordered_weights)


# ==============================================================================
# AGGREGATION OF PRIORITIES
# ==============================================================================

def aggregate_priorities(
    matrices: List[np.ndarray],
    method: str = "geometric_mean",
    expert_weights: List[float] | None = None,
    tolerance: float | None = None
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
        method (str): The method used to derive weights for each individual.
        expert_weights (List[float], optional): A list of weights for each expert's
                                                 final priority vector. If None,
                                                 equal weights are assumed.

    Returns:
        A single, aggregated crisp priority vector (NumPy array).
    """
    if not matrices:
        raise ValueError("Matrix list cannot be empty.")

    final_tolerance = tolerance if tolerance is not None else configure_parameters.FLOAT_TOLERANCE

    from .weight_derivation import derive_weights

    individual_crisp_weights = []
    for matrix in matrices:
        number_type = type(matrix[0, 0])
        results = derive_weights(matrix, number_type, method=method)
        individual_crisp_weights.append(results['crisp_weights'])

    weights_matrix = np.array(individual_crisp_weights)

    num_experts = len(matrices)
    if expert_weights is None:
        weights = np.full(num_experts, 1.0 / num_experts)
    else:
        if len(expert_weights) != num_experts:
            raise ValueError("Number of expert weights must match the number of matrices.")
        weight_sum = sum(expert_weights)
        if abs(weight_sum) < final_tolerance:
            raise ValueError("Sum of expert weights cannot be zero.")
        weights = np.array(expert_weights) / weight_sum

    final_group_priorities = np.average(weights_matrix, axis=0, weights=weights)

    return final_group_priorities / np.sum(final_group_priorities)

def aggregate_priorities_ifwa(
    priorities: List[IFN],
    expert_weights: List[float] | None = None,
    tolerance: float | None = None
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
        weights = [1.0 / num_priorities] * num_priorities
    else:
        if len(expert_weights) != num_priorities:
            raise ValueError(f"The number of expert weights provided ({len(expert_weights)}) "
                             f"must match the number of priorities ({num_priorities}).")

        final_tolerance = tolerance if tolerance is not None else configure_parameters.FLOAT_TOLERANCE

        weight_sum = sum(expert_weights)
        if abs(weight_sum) < final_tolerance:
             raise ValueError("The sum of expert weights cannot be zero.")

        weights = [w / weight_sum for w in expert_weights]

    # Formula for IFWA:
    # (1 - PRODUCT((1-mu_k)^w_k), PRODUCT(nu_k^w_k))
    prod_1_minus_mu = np.prod([(1 - p.mu) ** w for p, w in zip(priorities, weights)])
    prod_nu = np.prod([p.nu ** w for p, w in zip(priorities, weights)])

    agg_mu = 1 - prod_1_minus_mu
    agg_nu = prod_nu

    from .types import IFN
    return IFN(agg_mu, agg_nu)


def aggregate_priorities_tried(
    matrices: List[np.ndarray],
    derivation_method: str = "geometric_mean",
    expert_weights: List[float] | None = None,
    tolerance: float | None = None
) -> List[Number]:
    """
    Aggregates priorities by first calculating individual weight vectors for each
    expert's matrix and then combining these weight vectors.
    This corresponds to the "Aggregation of Priorities" workflow.

    .. note::
        This method is useful when you want to weigh the final calculated
        priorities of experts, rather than their initial judgments.

    Args:
        matrices: A list of comparison matrices from participants.
        method (str): The method used to derive weights for each individual.
        expert_weights (List[float], optional): A list of weights for each expert's
                                                 final priority vector. If None,
                                                 equal weights are assumed.

    Returns:
        A list of crisp priority vector (NumPy array).
    """
    if not matrices:
        raise ValueError("Matrix list cannot be empty.")

    from .weight_derivation import derive_weights
    number_type = type(matrices[0][0, 0])

    # 1. Calculate Individual Priority Vectors
    individual_fuzzy_weights = [] # Store FUZZY weight vectors
    individual_crisp_weights = [] # Store CRISP weight vectors

    for matrix in matrices:
        results = derive_weights(matrix, number_type, method=derivation_method)
        individual_fuzzy_weights.append(results['weights'])
        individual_crisp_weights.append(results['crisp_weights'])

    # 2. Aggregate based on number type
    if number_type.__name__ == 'IFN':
        # --- FUZZY AIP WORKFLOW for IFN ---
        # Transpose the list of fuzzy vectors
        weights_per_criterion = list(zip(*individual_fuzzy_weights))

        num_experts = len(matrices)
        if expert_weights is None:
            weights = [1.0 / num_experts] * num_experts
        else: # Normalize expert_weights
            weight_sum = sum(expert_weights)
            weights = [w / weight_sum for w in expert_weights]

        final_group_weights = []
        for criterion_weights in weights_per_criterion:
            # Aggregate the IFNs for one criterion from all experts using IFWA
            prod_1_minus_mu = np.prod([(1 - p.mu) ** w for p, w in zip(criterion_weights, weights)])
            prod_nu = np.prod([p.nu ** w for p, w in zip(criterion_weights, weights)])
            agg_mu = 1 - prod_1_minus_mu
            agg_nu = prod_nu
            final_group_weights.append(number_type(agg_mu, agg_nu))

        return final_group_weights # Return the final FUZZY weights

    else:
        # --- Standard CRISP aggregation for TFN, Crisp, etc. ---
        weights_matrix = np.array(individual_crisp_weights)
        # ... (your original logic to average the crisp weights)
        final_group_priorities = np.average(weights_matrix, axis=0, weights=weights)

        # Re-fuzzify the final crisp result
        return [number_type.from_normalized(p) for p in final_group_priorities]
