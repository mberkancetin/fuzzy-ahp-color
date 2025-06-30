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
    def _get_random_index(n: int) -> float:
        """Retrieves the Random Consistency Index (RI) from the global config."""
        return configure_parameters.SAATY_RI_VALUES.get(n, configure_parameters.SAATY_RI_VALUES['default'])

    @register_consistency_method("saaty_cr", has_threshold=True, threshold_func=_get_saaty_cr_threshold)
    def calculate_saaty_cr(matrix: np.ndarray, consistency_method: str = 'centroid', epsilon: float | None = None) -> float:
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

        Args:
            matrix: The comparison matrix of NumericType objects.
            consistency_method: The method used to convert fuzzy numbers to crisp ones.

        Returns:
            The approximate consistency ratio (CR) as a float.
        """
        n = matrix.shape[0]
        if n <= 2:
            return 0.0

        crisp_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cell = matrix[i, j]
                if hasattr(cell, 'defuzzify'):
                    crisp_matrix[i, j] = cell.defuzzify(method=consistency_method)
                else:
                    crisp_matrix[i, j] = float(cell)

        if np.any(crisp_matrix <= 0):
            raise ValueError("Comparison matrix contains non-positive values after defuzzification")

        # Calculate weights using numerically stable geometric mean
        log_matrix = np.log(crisp_matrix)
        row_geometric_means = np.exp(np.mean(log_matrix, axis=1))
        weights = row_geometric_means / np.sum(row_geometric_means)

        # Calculate lambda_max
        Aw = crisp_matrix @ weights

        if np.any(weights == 0):
            from .config import configure_parameters
            final_epsilon = epsilon if epsilon is not None else configure_parameters.LOG_EPSILON
            weights = np.maximum(weights, final_epsilon)

        lambda_max = np.mean(Aw / weights)

        ci = (lambda_max - n) / (n - 1)
        if ci < 0:
            ci = 0.0

        ri = Consistency._get_random_index(n)
        if ri <= 0:
            return 0.0

        return ci / ri

    @register_consistency_method("gci", has_threshold=True, threshold_func=_get_gci_threshold)
    def calculate_gci(matrix: np.ndarray, consistency_method: str = 'centroid') -> float:
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

        crisp_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cell = matrix[i, j]
                if hasattr(cell, 'defuzzify'):
                    crisp_matrix[i, j] = cell.defuzzify(method=consistency_method)
                else:
                    crisp_matrix[i, j] = float(cell)

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

    @register_consistency_method("mikhailov_lambda")
    def calculate_mikhailov_lambda(matrix: np.ndarray, consistency_method: str = 'centroid') -> float:
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
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Performs a comprehensive consistency check on all matrices in the model.

        This function dynamically runs all registered consistency methods and, for
        those with thresholds, determines a Pass/Fail status. The overall
        'is_consistent' flag is True only if all threshold-based checks pass.

        Args:
            model: The Hierarchy instance to check.
            **kwargs: Additional arguments to pass down to calculation functions,
                      e.g., `consistency_method='centroid'` or `saaty_cr_threshold=0.2`.

        Returns:
            A dictionary where keys are node IDs and values are a detailed
            dictionary of all calculated consistency metrics.
        """
        results_aggregator = {}

        def _recursive_check(node: Node):
            if node.comparison_matrix is None:
                return

            matrix = node.comparison_matrix
            n = matrix.shape[0]

            node_results = {"matrix_size": n}

            consistency_check_outcomes = []

            for name, meta in CONSISTENCY_METHODS.items():
                calc_func = meta["function"]
                try:
                    func_params = inspect.signature(calc_func).parameters

                    available_args = {
                        "matrix": matrix,
                        "number_type": type(matrix[0, 0]),
                    }
                    available_args.update(kwargs)

                    args_for_this_call = {
                        p_name: available_args[p_name]
                        for p_name in func_params
                        if p_name in available_args
                    }

                    value = calc_func(**args_for_this_call)
                    node_results[name] = value

                    if meta.get("has_threshold"):
                        threshold_func = meta["threshold_func"]

                        threshold_value = threshold_func(n, **kwargs)
                        node_results[f"{name}_threshold"] = threshold_value
                        final_tolerance = meta["tolerance"]

                        is_ok = False
                        status = "Not Calculated"
                        if isinstance(value, (int, float)):
                            is_ok = value <= (threshold_value + final_tolerance)
                            status = "Pass" if is_ok else "Fail"

                        node_results[f"{name}_status"] = status
                        consistency_check_outcomes.append(is_ok)

                except Exception as e:
                    node_results[name] = f"Error: {type(e).__name__} - {e}"
                    if meta.get("has_threshold"):
                        node_results[f"{name}_status"] = "Error"
                        consistency_check_outcomes.append(False)

            node_results["is_consistent"] = all(consistency_check_outcomes)
            results_aggregator[node.id] = node_results

        def traverse(node: Node):
            _recursive_check(node)
            for child in node.children:
                traverse(child)

        traverse(model.root)

        return results_aggregator

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
    def get_consistency_recommendations(model: Hierarchy, inconsistent_node_id: str, consistency_method: str = 'centroid') -> List[str]:
        """
        Provides specific recommendations for improving the consistency of a given matrix.
        """
        node = model._find_node(inconsistent_node_id)
        if node is None or node.comparison_matrix is None:
            return [f"Error: Node '{inconsistent_node_id}' not found or has no matrix."]

        matrix = node.comparison_matrix
        crisp_matrix = np.array([[cell.defuzzify(method=consistency_method) for cell in row] for row in matrix])
        n = matrix.shape[0]

        # Calculate weights from the crisp matrix using a robust method (geometric mean)
        # Add epsilon to avoid log(0)
        log_matrix = np.log(crisp_matrix + configure_parameters.LOG_EPSILON)
        weights = np.exp(np.mean(log_matrix, axis=1))
        weights /= np.sum(weights)

        # Find the judgment with the largest inconsistency
        max_error = -1
        inconsistent_pair = (None, None)
        for i in range(n):
            for j in range(i + 1, n):
                if weights[j] == 0: continue
                expected_val = weights[i] / weights[j]
                actual_val = crisp_matrix[i, j]
                # Logarithmic difference is better for ratio scales
                error = abs(np.log(actual_val) - np.log(expected_val))
                if error > max_error:
                    max_error = error
                    inconsistent_pair = (i, j)

        recommendations = []
        cr = Consistency.calculate_saaty_cr(matrix)
        recommendations.append(f"Matrix for node '{node.id}' is inconsistent (CR = {cr:.4f}).")

        if inconsistent_pair[0] is None:
            recommendations.append("Could not identify a primary inconsistent judgment.")
            return recommendations

        i, j = inconsistent_pair
        children_names = [child.id for child in node.children]

        recommendations.append(
            f"The most inconsistent judgment appears to be between '{children_names[i]}' and '{children_names[j]}'."
        )

        current_judgment = crisp_matrix[i, j]
        suggested_judgment = weights[i] / weights[j]
        recommendations.append(
            f"Your judgment was approximately {current_judgment:.2f}, but based on your other answers, it should be closer to {suggested_judgment:.2f}."
        )

        return recommendations
