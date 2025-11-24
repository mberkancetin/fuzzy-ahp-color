from __future__ import annotations
import inspect
from typing import Dict, List, Any, Type, TYPE_CHECKING, Callable
import numpy as np
from .config import configure_parameters

if TYPE_CHECKING:
    from .model import Hierarchy, Node
    from .types import NumericType, Number, TFN

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

def _check_pandas_availability():
    if not _PANDAS_AVAILABLE:
        raise ImportError("This feature requires the 'pandas' library. Please install it using: pip install pandas")


CONSISTENCY_METHODS: Dict[str, Callable] = {}

def register_consistency_method(
        name: str,
        has_threshold: bool = False,
        threshold_func: Callable | None = None,
        tolerance: float | None = None
    ):
    """
    A decorator to register a new consistency calculation method.

    Args:
        name: The name of the consistency index (e.g., 'saaty_cr').
        has_threshold: Set to True if this index has a pass/fail threshold.
        threshold_func: A function that takes matrix size `n` and returns the
                        threshold value. Required if `has_threshold` is True.
    """
    if has_threshold and threshold_func is None:
        raise ValueError(f"Method '{name}' is registered with a threshold but no threshold_func was provided.")

    def decorator(func: Callable) -> Callable:
        if name in CONSISTENCY_METHODS:
            print(f"Warning: Overwriting consistency method '{name}'")

        CONSISTENCY_METHODS[name] = {
            "function": func,
            "has_threshold": has_threshold,
            "threshold_func": threshold_func,
            "tolerance": tolerance if tolerance is not None else configure_parameters.FLOAT_TOLERANCE
        }
        return func
    return decorator


class Registry(dict):
    """
    A custom dictionary that validates insertions to ensure only
    callable objects (functions, methods) are registered.
    """
    def __setitem__(self, key: str, value: Callable):
        if not callable(value):
            # This is our guard rail. It will crash the program with a clear message.
            raise TypeError(
                f"Attempted to register a non-callable object of type '{type(value).__name__}' "
                f"for the key '{key}'. Only functions or methods can be registered."
            )
        super().__setitem__(key, value)

    def register(self, name: str) -> Callable:
        """Decorator factory for registering a function."""
        def decorator(func: Callable) -> Callable:
            self[name] = func
            return func
        return decorator

CONSISTENCY_METHODS = Registry()

class Consistency:
    """
    A class with static methods to calculate, check, and analyze the
    consistency of comparison matrices within an Hierarchy.
    """
    def _get_saaty_cr_threshold(n: int, saaty_cr_threshold: float | None = None, **kwargs) -> float:
        return saaty_cr_threshold if saaty_cr_threshold is not None else configure_parameters.DEFAULT_SAATY_CR_THRESHOLD

    def _get_gci_threshold(n: int, **kwargs) -> float:
        return configure_parameters.GCI_THRESHOLDS.get(n, configure_parameters.GCI_THRESHOLDS['default'])

    @staticmethod
    def _get_random_index(n: int, m: int = 0) -> float:
        """
        Retrieves the Random Consistency Index (RI) from the global config.
        If m > 0, it uses the generalized RI for incomplete matrices (Ágoston & Csató, 2022).

        Args:
            n: The size of the matrix.
            m: The number of missing entries above the diagonal. If 0,
               the standard Saaty RI is used.
        """
        if m > 0:
            # Try to find the exact generalized RI value
            val = configure_parameters.GENERALIZED_RI_VALUES.get((n, m))
            if val is not None:
                return val

            # If not found, use the linear approximation formula as a fallback
            if configure_parameters.USE_RI_APPROXIMATION_FALLBACK and n > 2:
                ri_complete = configure_parameters.SAATY_RI_VALUES.get(n, configure_parameters.SAATY_RI_VALUES['default'])
                # The formula from the paper is:
                # RI_n,m ≈ (1 - 2m / ((n-1)(n-2))) * RI_n,0
                denominator = (n - 1) * (n - 2)
                if denominator > 0:
                    return (1 - (2 * m) / denominator) * ri_complete

        return configure_parameters.SAATY_RI_VALUES.get(n, configure_parameters.SAATY_RI_VALUES['default'])

    @CONSISTENCY_METHODS.register("saaty_cr")
    def calculate_saaty_cr(
        matrix: np.ndarray,
        consistency_method: str = 'centroid',
        epsilon: float | None = None,
        num_missing_pairs: int = 0, **kwargs) -> float:
        """
        Calculates Saaty's traditional Consistency Ratio (CR) by first
        defuzzifying the fuzzy matrix. This is a practical approximation
        using a numerically stable geometric mean method.

        .. warning::
            **Academic Limitation:** This is a widely used practical approximation. True
            consistency of a fuzzy matrix is a complex topic in academic literature,
            as traditional CR was designed for crisp values and does not account for
            the uncertainty inherent in fuzzy numbers. This method should be used as a
            heuristic to check for major inconsistencies in the defuzzified judgments.
            For rigorous academic work, consider citing literature on fuzzy consistency
            measures (e.g., based on alpha-cuts or fuzzy consistency indices).

        .. note::
            If `num_missing_pairs` > 0, this method uses the generalized Random Index
            to account for the artificial consistency introduced by optimization-based
            matrix completion.

        Args:
            matrix: The comparison matrix of NumericType objects.
            consistency_method: The method used to convert fuzzy numbers to crisp ones.

        Returns:
            The approximate consistency ratio (CR) as a float.
        """
        n = matrix.shape[0]
        if n <= 2:
            return 0.0

        first_element = matrix[0, 0]
        if hasattr(first_element, 'defuzzify'):
            crisp_matrix = np.array([[cell.defuzzify(method=consistency_method) for cell in row] for row in matrix], dtype=float)
        elif isinstance(first_element, (int, float, np.number)):
            crisp_matrix = matrix.astype(float)
        else:
            raise TypeError(f"Unsupported matrix element type for consistency check: {type(first_element)}")

        if np.any(crisp_matrix <= 0):
            return np.inf
        # Calculate weights using numerically stable geometric mean
        log_matrix = np.log(crisp_matrix)
        row_geometric_means = np.exp(np.mean(log_matrix, axis=1))
        weights = row_geometric_means / np.sum(row_geometric_means)

        # Calculate lambda_max
        Aw = crisp_matrix @ weights

        if np.any(weights == 0):
            final_epsilon = epsilon if epsilon is not None else configure_parameters.LOG_EPSILON
            weights = np.maximum(weights, final_epsilon)

        lambda_max = np.mean(Aw / weights)

        ci = (lambda_max - n) / (n - 1)
        if ci < 0:
            ci = 0.0

        ri = Consistency._get_random_index(n, num_missing_pairs)

        if ri <= 0:
            return float('inf') if ci > 0 else 0.0 # Avoid division by zero

        return ci / ri

    @CONSISTENCY_METHODS.register("gci")
    def calculate_gci(matrix: np.ndarray, consistency_method: str = 'centroid', **kwargs) -> float:
        """
        Calculates the Geometric Consistency Index (GCI) for the matrix.
        A lower GCI value indicates better consistency.

        .. note::
            **Academic Note:** GCI is an alternative to Saaty's CR. Thresholds
            proposed by Aguarón & Moreno-Jiménez (2003) are often cited:
            GCI <= 0.31 for n=3, <= 0.35 for n=4, <= 0.37 for n>4.
        """
        n = matrix.shape[0]
        if n <= 2: return 0.0

        first_element = matrix[0, 0]
        if hasattr(first_element, 'defuzzify'):
            crisp_matrix = np.array([[cell.defuzzify(method=consistency_method) for cell in row] for row in matrix], dtype=float)
        elif isinstance(first_element, (int, float, np.number)):
            crisp_matrix = matrix.astype(float)
        else:
            raise TypeError(f"Unsupported matrix element type for consistency check: {type(first_element)}")


        if np.any(crisp_matrix <= 0): return np.inf

        # Calculate weights using the geometric mean method
        log_matrix = np.log(crisp_matrix)
        weights = np.exp(np.mean(log_matrix, axis=1))
        weights /= np.sum(weights)

        # Calculate GCI using the corrected formula
        sum_of_squared_errors = 0
        for i in range(n):
            for j in range(i + 1, n): # Iterate through upper triangle
                error = np.log(crisp_matrix[i, j]) - np.log(weights[i]) + np.log(weights[j])
                sum_of_squared_errors += error**2

        gci = (2 / ((n - 1) * (n - 2))) * sum_of_squared_errors
        return gci

    @CONSISTENCY_METHODS.register("mikhailov_lambda")
    def calculate_mikhailov_lambda(matrix: np.ndarray, consistency_method: str = 'centroid', **kwargs) -> float:
        """
        Calculates the consistency index (lambda) using Mikhailov's (2004)
        Fuzzy Programming method.

        .. note::
            **Academic Note:** This is a true fuzzy consistency measure.
            A value of λ > 0 indicates that a fully consistent crisp weight
            vector exists within the bounds of the fuzzy judgments.
            Values closer to 1 indicate higher consistency. A value of λ < 0
            indicates inconsistency.
        """
        from .weight_derivation import mikhailov_fuzzy_programming
        from .types import TFN
        if not isinstance(matrix[0,0], TFN):
            return "N/A (TFN only)"

        number_type = type(matrix[0,0])

        try:
            results = mikhailov_fuzzy_programming(matrix, number_type)
            if results['optimization_success']:
                return results['lambda_consistency']
            else:
                message = results.get('optimization_message', 'Optimization failed.')
                reason = message.split(':')[1].strip() if ':' in message else message
                return f"Failed ({reason})"
        except Exception as e:
            return f"Error ({type(e).__name__})"

    @staticmethod
    def check_model_consistency(
        model: Hierarchy,
        consistency_method: str = 'centroid',
        saaty_cr_threshold: float | None = None,
        required_indices: List[str] | None = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Performs a comprehensive, configurable consistency check on all matrices in the model.

        This method traverses the hierarchy and runs all registered consistency algorithms
        (e.g., 'saaty_cr', 'gci') on each comparison matrix. It then determines an overall
        'is_consistent' status based on a flexible set of requirements.

        .. rubric:: Academic Description

        The concept of a single, universally accepted measure of matrix consistency is
        a subject of ongoing academic debate. Saaty's (1980) Consistency Ratio (CR) is
        the most widely known, but alternatives such as the Geometric Consistency Index (GCI)
        by Aguarón & Moreno-Jiménez (2003) are often preferred for their mathematical
        properties, especially when using the geometric mean for weight derivation.

        This function embraces this diversity by adopting a modular approach. It calculates
        all available consistency indices registered in the library. Crucially, it allows
        the researcher to define what constitutes "consistency" for their specific study via
        the `required_indices` parameter. This enables various research scenarios:
        - Replicating studies that rely solely on Saaty's CR.
        - Performing analyses where only the GCI is considered relevant.
        - Defining a stricter consistency requirement where multiple indices must all
        pass their respective thresholds.

        By default, if no indices are specified, the function adheres to a common modern
        practice of checking both Saaty's CR and the GCI. A matrix is only flagged as
        `is_consistent: True` if all required numerical indices are below their established
        thresholds and no registered index calculation results in a failure.

        Args:
            model (Hierarchy): The Hierarchy instance to check.
            consistency_method (str, optional): The defuzzification method used for
                indices that require a crisp matrix. Defaults to 'centroid'.
            saaty_cr_threshold (float, optional): The acceptable threshold for Saaty's CR.
                If None, it uses the value from the global `configure_parameters`.
            required_indices (List[str], optional): A list of consistency index names
                that MUST pass for a matrix to be considered consistent. If None, it
                defaults to checking all primary numerical indices available (typically
                'saaty_cr' and 'gci').

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are node IDs and values
            are a detailed dictionary of all calculated consistency metrics and the
            overall `is_consistent` status.
        """
        final_cr_threshold = saaty_cr_threshold if saaty_cr_threshold is not None else configure_parameters.DEFAULT_SAATY_CR_THRESHOLD

        results = {}

        def _recursive_check(node: Node):
            for child in node.children:
                _recursive_check(child)

            if node.comparison_matrix is None:
                return

            matrix = node.comparison_matrix
            n = matrix.shape[0]
            number_type = type(matrix[0, 0])

            num_missing_pairs = np.count_nonzero(matrix == None) // 2

            node_results = {
                "matrix_size": n,
                "num_missing_pairs": num_missing_pairs
            }

            context_args = {
                "matrix": matrix,
                "number_type": number_type,
                "consistency_method": consistency_method,
                "num_missing_pairs": num_missing_pairs,
                "model": model
            }

            for name, func in CONSISTENCY_METHODS.items():
                try:
                    node_results[name] = func(**context_args)
                except Exception as e:
                    node_results[name] = f"Error: {e}"

            indices_to_check = required_indices
            if indices_to_check is None:
                indices_to_check = []
                if "saaty_cr" in CONSISTENCY_METHODS: indices_to_check.append("saaty_cr")
                if "gci" in CONSISTENCY_METHODS: indices_to_check.append("gci")

            all_required_passed = True

            for index_name in indices_to_check:
                value = node_results.get(index_name)
                if isinstance(value, str) and value.lower().startswith(('error', 'failed')):
                    all_required_passed = False
                    break

            if all_required_passed:
                if "saaty_cr" in indices_to_check:
                    saaty_cr_val = node_results.get("saaty_cr")
                    if not (isinstance(saaty_cr_val, (float, np.floating)) and saaty_cr_val <= final_cr_threshold):
                        all_required_passed = False

                if "gci" in indices_to_check:
                    gci_val = node_results.get("gci")
                    gci_threshold = Consistency._get_gci_threshold(n)
                    if not (isinstance(gci_val, (float, np.floating)) and gci_val <= gci_threshold):
                        all_required_passed = False

            node_results["is_consistent"] = all_required_passed
            node_results["saaty_cr_threshold"] = final_cr_threshold
            node_results["gci_threshold"] = Consistency._get_gci_threshold(n) # Report for context

            results[node.id] = node_results

        _recursive_check(model.root)
        return results

    @staticmethod
    def check_group_consistency(
        model_template: Hierarchy,
        expert_matrices: Dict[str, List[np.ndarray]],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Checks the consistency for each expert's set of matrices individually.

        This is primarily used for the 'Aggregation of Individual Priorities' (AIP)
        workflow, where each expert's consistency should be assessed before
        aggregating their final priorities.

        Args:
            model_template: A template Hierarchy object (with structure but no data)
                            to be copied for each expert.
            expert_matrices: A dictionary mapping node IDs to a list of matrices,
                             one matrix per expert.
            **kwargs: Additional arguments to pass to check_model_consistency
                      (e.g., saaty_cr_threshold).

        Returns:
            A pandas DataFrame summarizing the consistency results for each
            expert and each of their matrices.
        """
        _check_pandas_availability()

        num_experts = len(list(expert_matrices.values())[0])
        all_results = []

        for i in range(num_experts):
            from copy import deepcopy
            expert_model = deepcopy(model_template)

            for node_id, matrices in expert_matrices.items():
                node = expert_model._find_node(node_id)
                if not node: continue

                if node.is_leaf:
                    expert_model.set_alternative_matrix(node_id, matrices[i])
                else:
                    expert_model.set_comparison_matrix(node_id, matrices[i])

            consistency_report = Consistency.check_model_consistency(expert_model, **kwargs)

            for node_id, metrics in consistency_report.items():
                record = {
                    "Expert": f"Expert {i+1}",
                    "Node": node_id,
                    "CR": metrics.get("saaty_cr"),
                    "CR Status": metrics.get("saaty_cr_status"),
                    "GCI": metrics.get("gci"),
                    "GCI Status": metrics.get("gci_check_status"),
                    "Is Consistent": metrics.get("is_consistent")
                }
                all_results.append(record)

        if not all_results:
            return {"Info": "No matrices were provided to check consistency."}

        all_results['CR'] = [f"{x:.4f}" if isinstance(x, float) else x for x in all_results['CR']]
        all_results['GCI'] = [f"{x:.4f}" if isinstance(x, float) else x for x in all_results['GCI']]

        return all_results

    @staticmethod
    def get_consistency_recommendations(
        model: Hierarchy,
        inconsistent_node_id: str,
        consistency_method: str = 'centroid',
        recommendation_method: str = "logarithmic_error_magnitude"
    ) -> Dict[str, Any]:
        """
        Provides a ranked list of judgments to change to improve consistency.
        """
        node = model._find_node(inconsistent_node_id)
        if node is None or node.comparison_matrix is None:
            return {'error': f"Node '{inconsistent_node_id}' not found or has no matrix."}

        matrix = node.comparison_matrix
        crisp_matrix = np.array([[c.defuzzify(method=consistency_method) for c in row] for row in matrix], dtype=float)
        n = crisp_matrix.shape[0]

        first_element = matrix[0, 0]
        if hasattr(first_element, 'defuzzify'):
            crisp_matrix = np.array([[cell.defuzzify(method=consistency_method) for cell in row] for row in matrix], dtype=float)
        elif isinstance(first_element, (int, float, np.number)):
            crisp_matrix = matrix.astype(float)
        else:
            raise TypeError(f"Unsupported matrix element type for consistency check: {type(first_element)}")


        sanitized_matrix = np.maximum(crisp_matrix, 1e-9)

        try:
            log_matrix = np.log(sanitized_matrix)
            weights = np.exp(np.mean(log_matrix, axis=1))

            weight_sum = np.sum(weights)
            if weight_sum < 1e-9:
                raise ValueError("Sum of weights is near zero.")

            weights /= weight_sum
        except Exception as e:
            return {'error': f"Failed to calculate weights for recommendation. Error: {e}"}

        all_errors = []
        for i in range(n):
            for j in range(i + 1, n):
                if weights[j] < 1e-9:
                    continue

                expected_val = weights[i] / weights[j]
                actual_val = crisp_matrix[i, j]

                if recommendation_method == "relative_error":
                    if abs(expected_val) > 1e-9:
                        error = abs(actual_val - expected_val) / expected_val
                    else:
                        error = abs(actual_val - expected_val)
                else:
                    # Calculate the ratio of the actual value to the expected value
                    # This is the core of the geometric consistency error (GCI is based on log of this ratio)
                    # A value of 1 means perfect consistency for this pair.
                    ratio_error = actual_val / expected_val

                    # The magnitude of the log of the ratio is a more robust measure of inconsistency
                    # for ratio-scale data like AHP.
                    error_magnitude = abs(np.log(ratio_error))

                    # We use the error_magnitude to rank the pairs.
                    error = error_magnitude


                all_errors.append({
                    "pair": (i, j),
                    "error": error,
                    "current_value": actual_val,
                    "suggested_value": expected_val,
                })

        if not all_errors:
            return {'error': "Could not calculate inconsistency errors for any pair."}

        all_errors.sort(key=lambda x: x['error'], reverse=True)

        return {
            "revisions": all_errors,
            "children_names": [child.id for child in node.children]
        }
