from __future__ import annotations
from typing import Dict, List, Any, Type, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .model import Hierarchy, Node
    from .types import NumericType, Number, TFN


class Consistency:
    """
    A class with static methods to calculate, check, and analyze the
    consistency of comparison matrices within an Hierarchy.
    """

    # Saaty's Random Consistency Index (RI) values
    # Source: Saaty, T. L. (2008). Relative measurement and its generalization in
    # decision making why pairwise comparisons are central in mathematics...
    # RACSAM-Revista de la Real Academia de Ciencias Exactas, Fisicas y Naturales.
    _RI_VALUES = {
        1: 0.00, 2: 0.00, 3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35,
        8: 1.40, 9: 1.45, 10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59
    }

    _GCI_THRESHOLDS = {3: 0.31, 4: 0.35} # Default for n>4 is 0.37

    @staticmethod
    def _get_random_index(n: int) -> float:
        """Retrieves the Random Consistency Index (RI) for a given matrix size."""
        return Consistency._RI_VALUES.get(n, 1.60) # Default for n > 15

    @staticmethod
    def _get_gci_threshold(n: int) -> float:
        return Consistency._GCI_THRESHOLDS.get(n, 0.37)

    @staticmethod
    def calculate_saaty_cr(matrix: np.ndarray, consistency_method: str = 'centroid') -> float:
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

        # Defuzzify the matrix
        crisp_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cell = matrix[i, j]
                if hasattr(cell, 'defuzzify'):
                    crisp_matrix[i, j] = cell.defuzzify(method=consistency_method)
                else:
                    crisp_matrix[i, j] = float(cell)

        # Validate matrix values
        if np.any(crisp_matrix <= 0):
            raise ValueError("Comparison matrix contains non-positive values after defuzzification")

        # Calculate weights using numerically stable geometric mean
        log_matrix = np.log(crisp_matrix)
        row_geometric_means = np.exp(np.mean(log_matrix, axis=1))
        weights = row_geometric_means / np.sum(row_geometric_means)

        # Calculate lambda_max
        Aw = crisp_matrix @ weights

        # Handle potential zero weights more robustly
        if np.any(weights == 0):
            # This shouldn't happen with proper geometric mean calculation
            weights = np.maximum(weights, 1e-10)

        lambda_max = np.mean(Aw / weights)

        # Calculate CI
        ci = (lambda_max - n) / (n - 1)

        # CI should theoretically be non-negative; negative values indicate numerical errors
        if ci < 0:
            ci = 0.0

        # Get Random Index and calculate CR
        ri = Consistency._get_random_index(n)

        if ri <= 0:
            return 0.0

        return ci / ri

    @staticmethod
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

    @staticmethod
    def calculate_mikhailov_lambda(matrix: np.ndarray, number_type: Type[TFN]) -> float:
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

        try:
            results = mikhailov_fuzzy_programming(matrix, number_type)
            if results['optimization_success']:
                return results['lambda_consistency']
            else:
                return -1.0 # Return negative value to indicate optimization failure
        except Exception:
            return -1.0

    @staticmethod
    def check_model_consistency(
        model: Hierarchy,
        consistency_method: str = 'centroid',
        saaty_cr_threshold: float = 0.1
    ) -> Dict[str, Dict[str, Any]]:
        """
        Performs a comprehensive consistency check on all matrices in the model,
        calculating Saaty's CR, GCI, and Mikhailov's Lambda where applicable.

        Args:
            model: The Hierarchy (AHPModel) instance to check.
            consistency_method: The defuzzification method used for CR and GCI.
            saaty_cr_threshold: The acceptable threshold for Saaty's CR.

        Returns:
            A dictionary where keys are node IDs and values are a detailed
            dictionary of all calculated consistency metrics.
        """
        results = {}
        from .types import TFN
        def _recursive_check(node: Node):
            if node.comparison_matrix is not None:
                matrix = node.comparison_matrix
                n = matrix.shape[0]
                node_results = {"matrix_size": n}

                # --- Calculate all relevant metrics ---
                # 1. Saaty's CR (Approximation)
                saaty_cr = Consistency.calculate_saaty_cr(matrix, consistency_method)
                node_results["saaty_cr"] = saaty_cr

                # 2. Geometric Consistency Index (GCI)
                gci = Consistency.calculate_gci(matrix, consistency_method)
                node_results["gci"] = gci

                # 3. Mikhailov's Lambda (only for TFN matrices)
                if isinstance(matrix[0,0], TFN):
                    lambda_consistency = Consistency.calculate_mikhailov_lambda(matrix, TFN)
                    node_results["mikhailov_lambda"] = lambda_consistency
                else:
                    node_results["mikhailov_lambda"] = "N/A"

                # --- Determine overall consistency status ---
                gci_threshold = Consistency._get_gci_threshold(n)

                is_cr_ok = saaty_cr <= saaty_cr_threshold
                is_gci_ok = gci <= gci_threshold

                # Define overall consistency. We can say it's consistent if both
                # major indices are below their respective thresholds.
                node_results["is_consistent"] = is_cr_ok and is_gci_ok
                node_results["saaty_cr_threshold"] = saaty_cr_threshold
                node_results["gci_threshold"] = gci_threshold

                results[node.id] = node_results

            for child in node.children:
                _recursive_check(child)

        _recursive_check(model.root)
        return results

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
        log_matrix = np.log(crisp_matrix + 1e-10)
        weights = np.exp(np.mean(log_matrix, axis=1))
        weights /= np.sum(weights)

        # Find the judgment with the largest inconsistency
        max_error = -1
        inconsistent_pair = (None, None)
        for i in range(n):
            for j in range(i + 1, n):
                if weights[j] == 0: continue # Avoid division by zero
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

        # Line 2: Identify the pair
        recommendations.append(
            f"The most inconsistent judgment appears to be between '{children_names[i]}' and '{children_names[j]}'."
        )

        # Line 3: Provide the values
        current_judgment = crisp_matrix[i, j]
        suggested_judgment = weights[i] / weights[j]
        recommendations.append(
            f"Your judgment was approximately {current_judgment:.2f}, but based on your other answers, it should be closer to {suggested_judgment:.2f}."
        )

        return recommendations
