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
        if m > 0 and n > 2:
            ri_complete = configure_parameters.SAATY_RI_VALUES.get(n, 1.6)
            denominator = (n - 1) * (n - 2)
            if denominator > 0:
                return (1 - (2 * m) / denominator) * ri_complete
        return configure_parameters.SAATY_RI_VALUES.get(n, 1.6)

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
        if n <= 2: return 0.0

        # Create a strictly reciprocal crisp matrix
        crisp_matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                val = matrix[i, j].defuzzify(method=consistency_method)
                if val <= 1e-9: val = 1e-9

                crisp_matrix[i, j] = val
                crisp_matrix[j, i] = 1.0 / val

        # Geometric mean weights
        log_matrix = np.log(crisp_matrix)
        row_geometric_means = np.exp(np.mean(log_matrix, axis=1))
        weights = row_geometric_means / np.sum(row_geometric_means)

        # Lambda max
        Aw = crisp_matrix @ weights
        lambda_max = np.mean(Aw / np.maximum(weights, 1e-9))

        ci = (lambda_max - n) / (n - 1)
        if ci < 0: ci = 0.0

        ri = Consistency._get_random_index(n, num_missing_pairs)
        if ri <= 0: return float('inf') if ci > 0 else 0.0

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

        crisp_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                val = matrix[i, j].defuzzify(method=consistency_method)
                if val <= 1e-9: val = 1e-9
                crisp_matrix[i, j] = val
                crisp_matrix[j, i] = 1.0 / val

        log_matrix = np.log(crisp_matrix)
        weights = np.exp(np.mean(log_matrix, axis=1))
        weights /= np.sum(weights)

        sum_of_squared_errors = 0
        for i in range(n):
            for j in range(i + 1, n):
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
        final_cr_threshold = saaty_cr_threshold or configure_parameters.DEFAULT_SAATY_CR_THRESHOLD
        results = {}

        def _recursive_check(node):
            # Check consistency for ANY node that has a matrix (Internal or Leaf)
            if node.comparison_matrix is not None:
                matrix = node.comparison_matrix
                n = matrix.shape[0]
                node_results = {"matrix_size": n}

                for name, func in CONSISTENCY_METHODS.items():
                    try:
                        node_results[name] = func(matrix=matrix, consistency_method=consistency_method, num_missing_pairs=0)
                    except Exception as e:
                        node_results[name] = f"Error: {e}"

                indices = required_indices or ["saaty_cr", "gci"]
                consistent = True

                if "saaty_cr" in indices:
                    val = node_results.get("saaty_cr", 99)
                    if isinstance(val, (float, np.floating)) and val > final_cr_threshold:
                        consistent = False

                if "gci" in indices:
                    val = node_results.get("gci", 99)
                    thresh = Consistency._get_gci_threshold(n)
                    if isinstance(val, (float, np.floating)) and val > thresh:
                        consistent = False

                node_results["is_consistent"] = consistent
                results[node.id] = node_results

            for child in node.children:
                _recursive_check(child)

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
        if not node or node.comparison_matrix is None: return {'error': 'Node not found or has no matrix'}

        matrix = node.comparison_matrix
        n = matrix.shape[0]

        crisp = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                val = matrix[i, j].defuzzify(method=consistency_method)
                if val <= 1e-9: val = 1e-9
                crisp[i, j] = val
                crisp[j, i] = 1.0/val

        try:
            log_m = np.log(crisp)
            w = np.exp(np.mean(log_m, axis=1))
            w /= np.sum(w)
        except Exception as e:
            return {'error': f"Weight calc failed: {e}"}

        all_errors = []
        for i in range(n):
            for j in range(i+1, n):
                expected = w[i] / w[j]
                actual = crisp[i, j]
                error = abs(np.log(actual / expected))
                all_errors.append({
                    "pair": (i, j),
                    "error": error,
                    "current_value": actual,
                    "suggested_value": expected
                })

        # Deterministic sort
        all_errors.sort(key=lambda x: (-x['error'], x['pair']))

        children_names = [child.id for child in node.children]

        # If node is a leaf (Alternative Matrix), names are Alternatives, not Children
        if not children_names and model.alternatives:
             # Assuming standard order of alternatives in matrix
             children_names = [alt.name for alt in model.alternatives]

        return {
            "revisions": all_errors,
            "children_names": children_names
        }
