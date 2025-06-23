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
        Calculates the Consistency Ratio (CR) for a single comparison matrix.
        This function works for any matrix of NumericType objects.

        Args:
            matrix: The comparison matrix.
            defuzzify_method: The method used to convert fuzzy numbers to crisp ones.

        Returns:
            The consistency ratio (CR) as a float.
        """
        n = matrix.shape[0]
        if n <= 2:
            return 0.0

        # 1. Defuzzify the matrix to get a crisp matrix for calculation.
        crisp_matrix = np.array([[cell.defuzzify(method=defuzzify_method) for cell in row] for row in matrix])

        # 2. Calculate the principal eigenvalue (lambda_max).
        try:
            eigenvalues, _ = np.linalg.eig(crisp_matrix)
            lambda_max = np.max(np.real(eigenvalues))
        except np.linalg.LinAlgError:
            # Fallback if eigenvalue computation fails
            # Use the approximation from the weights
            col_sums = np.sum(crisp_matrix, axis=0)
            norm_matrix = crisp_matrix / col_sums
            weights = np.mean(norm_matrix, axis=1)
            lambda_max = np.mean(np.dot(crisp_matrix, weights) / weights)

        # 3. Calculate Consistency Index (CI).
        # Add a check to prevent division by zero if n=1 (though already handled)
        if n <= 1: return 0.0
        ci = (lambda_max - n) / (n - 1)

        # 4. Calculate Consistency Ratio (CR).
        ri = Consistency._get_random_index(n)
        return ci / ri if ri > 0 else 0.0

    @staticmethod
    def check_model_consistency(model: Hierarchy, defuzzify_method: str = 'centroid', threshold: float = 0.1) -> Dict[str, Dict[str, Any]]:
        """
        Checks the consistency of all comparison matrices in the entire AHP model.

        Args:
            model: The Hierarchy instance to check.
            defuzzify_method: The defuzzification method to use.
            threshold: The acceptable CR threshold (e.g., 0.1 for matrices > 4x4).

        Returns:
            A dictionary where keys are node IDs and values are detailed consistency results.
        """
        results = {}

        def _recursive_check(node: 'Node'):
            if not node.is_leaf and node.comparison_matrix is not None:
                n = node.comparison_matrix.shape[0]
                cr = Consistency.calculate_consistency_ratio(node.comparison_matrix, defuzzify_method)

                # Use a more lenient threshold for small matrices, as Saaty suggested.
                current_threshold = 0.05 if n == 3 else 0.08 if n == 4 else threshold

                results[node.id] = {
                    "is_consistent": cr <= current_threshold,
                    "consistency_ratio": cr,
                    "matrix_size": n,
                    "threshold": current_threshold
                }

            for child in node.children:
                _recursive_check(child)

        _recursive_check(model.root)
        return results

    @staticmethod
    def get_consistency_recommendations(model: Hierarchy, inconsistent_node_id: str) -> List[str]:
        """
        Provides specific recommendations for improving the consistency of a given matrix
        within the AHP model.

        Args:
            model: The Hierarchy instance.
            inconsistent_node_id: The ID of the parent node whose matrix is inconsistent.

        Returns:
            A list of human-readable recommendations.
        """
        node = model._find_node(inconsistent_node_id)
        if node is None or node.comparison_matrix is None:
            return [f"Error: Node '{inconsistent_node_id}' not found or has no matrix."]

        matrix = node.comparison_matrix
        crisp_matrix = np.array([[cell.defuzzify() for cell in row] for row in matrix])
        n = matrix.shape[0]

        # Calculate weights from the crisp matrix (using geometric mean as a robust estimator)
        weights = np.array([np.prod(row)**(1.0/n) for row in crisp_matrix])
        weights /= np.sum(weights)

        # Find the most inconsistent judgment
        max_error = -1
        inconsistent_pair = (None, None)
        for i in range(n):
            for j in range(i + 1, n):
                expected_val = weights[i] / weights[j]
                actual_val = crisp_matrix[i, j]
                # Use logarithmic difference for ratio scale error
                error = abs(np.log(actual_val) - np.log(expected_val))
                if error > max_error:
                    max_error = error
                    inconsistent_pair = (i, j)

        if inconsistent_pair[0] is None:
            return ["Could not identify a primary inconsistent judgment."]

        i, j = inconsistent_pair
        children_names = [child.id for child in node.children]

        recommendations = []
        cr = Consistency.calculate_consistency_ratio(matrix)
        recommendations.append(f"Matrix for node '{node.id}' is inconsistent (CR = {cr:.4f}).")

        recommendations.append(
            f"The most inconsistent judgment appears to be between "
            f"'{children_names[i]}' and '{children_names[j]}'."
        )

        current_judgment = crisp_matrix[i, j]
        suggested_judgment = weights[i] / weights[j]

        recommendations.append(
            f"Your judgment was approximately {current_judgment:.2f}, but based on your other answers, "
            f"it should be closer to {suggested_judgment:.2f}."
        )

        if current_judgment > suggested_judgment:
            recommendations.append(
                f"Suggestion: Re-evaluate if '{children_names[i]}' is truly that much more important "
                f"than '{children_names[j]}'."
            )
        else:
            recommendations.append(
                f"Suggestion: Re-evaluate if '{children_names[j]}' is truly that much more important "
                f"than '{children_names[i]}'."
            )

        return recommendations
