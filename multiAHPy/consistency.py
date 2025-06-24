from __future__ import annotations
from typing import Dict, List, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from multiAHPy.model import Hierarchy, Node
    from multiAHPy.types import NumericType, Number, TFN

class Consistency:
    """
    A class with static methods to calculate, check, and analyze the
    consistency of comparison matrices within an Hierarchy.
    """

    # Saaty's Random Consistency Index (RI) values
    _RI_VALUES = {
        1: 0.00, 2: 0.00, 3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35,
        8: 1.40, 9: 1.45, 10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59
    }

    @staticmethod
    def _get_random_index(n: int) -> float:
        """Retrieves the Random Consistency Index (RI) for a given matrix size."""
        return Consistency._RI_VALUES.get(n, 1.60) # Default for n > 15

    @staticmethod
    def calculate_consistency_ratio(matrix: np.ndarray, defuzzify_method: str = 'centroid') -> float:
        """
        Calculates an approximate Consistency Ratio (CR) for a fuzzy comparison matrix
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
            defuzzify_method: The method used to convert fuzzy numbers to crisp ones.

        Returns:
            The approximate consistency ratio (CR) as a float.
        """
        n = matrix.shape[0]
        if n <= 2:
            return 0.0

        # Defuzzify the matrix
        crisp_matrix = np.array([[cell.defuzzify(method=defuzzify_method) for cell in row] for row in matrix])

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
    def check_model_consistency(model: Hierarchy, defuzzify_method: str = 'centroid', threshold: float = 0.1) -> Dict[str, Dict[str, Any]]:
        """
        Checks the consistency of all comparison matrices in the entire AHP model.
        """
        results = {}

        def _recursive_check(node: 'Node'):
            if not node.is_leaf and node.comparison_matrix is not None:
                n = node.comparison_matrix.shape[0]
                cr = Consistency.calculate_consistency_ratio(node.comparison_matrix, defuzzify_method)
                is_consistent = cr <= threshold

                results[node.id] = {
                    "is_consistent": is_consistent,
                    "consistency_ratio": cr,
                    "matrix_size": n,
                    "threshold": threshold
                }

            for child in node.children:
                _recursive_check(child)

        _recursive_check(model.root)
        return results

    @staticmethod
    def get_consistency_recommendations(model: Hierarchy, inconsistent_node_id: str, defuzzify_method: str = 'centroid') -> List[str]:
        """
        Provides specific recommendations for improving the consistency of a given matrix.
        """
        node = model._find_node(inconsistent_node_id)
        if node is None or node.comparison_matrix is None:
            return [f"Error: Node '{inconsistent_node_id}' not found or has no matrix."]

        matrix = node.comparison_matrix
        crisp_matrix = np.array([[cell.defuzzify(method=defuzzify_method) for cell in row] for row in matrix])
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
        cr = Consistency.calculate_consistency_ratio(matrix)
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
