from __future__ import annotations
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from multiAHPy.model import Hierarchy, Node
    from multiAHPy.types import TFN, TrFN, Crisp, GFN, NumericType, Number

class Validation:
    """
    A class containing static methods to validate an Hierarchy and its components.
    including individual comparison matrices.
    """

    @staticmethod
    def validate_matrix_properties(matrix: np.ndarray, tolerance: float = 1e-6) -> List[str]:
        """
        Validates a single comparison matrix for dimensions, diagonal, and reciprocity.

        Args:
            matrix: The comparison matrix to validate.
            tolerance: Tolerance for floating-point reciprocity checks.

        Returns:
            A list of error strings. An empty list means the matrix is valid.
        """
        errors = []

        # 1. Validate Dimensions
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            errors.append("Matrix must be a 2D square NumPy array.")
            return errors # Stop further checks if dimensions are wrong

        n = matrix.shape[0]
        if n < 2:
            errors.append("Matrix must be at least 2x2.")

        # 2. Validate Diagonal
        # We check the defuzzified centroid value for the diagonal.
        identity_one = matrix[0,0].multiplicative_identity()
        for i in range(n):
            if abs(matrix[i, i].defuzzify() - identity_one.defuzzify()) > tolerance:
                errors.append(f"Diagonal element at ({i},{i}) is not 1. Found: {matrix[i,i]}")

        # 3. Validate Reciprocity
        for i in range(n):
            for j in range(i + 1, n):
                inverse_val = matrix[j, i].inverse()
                val = matrix[i, j]

                # Compare the centroids of the two fuzzy numbers
                if abs(val.defuzzify() - inverse_val.defuzzify()) > tolerance:
                    errors.append(f"Reciprocity failed between ({i},{j}) and ({j},{i}). "
                                  f"Value: {val}, Inverse of counterpart: {inverse_val}")

        return errors

    @staticmethod
    def validate_hierarchy_completeness(model: Hierarchy) -> List[str]:
        """
        Recursively validates the completeness of the AHP hierarchy structure.
        Checks that every parent node has a comparison matrix defined for its children.

        Args:
            model: The Hierarchy instance to validate.

        Returns:
            A list of error strings describing missing matrices.
        """
        errors = []

        def _recursive_check(node: 'Node'):
            # A node is a parent if it's not a leaf.
            if not node.is_leaf:
                # Check if this parent has a comparison matrix.
                if node.comparison_matrix is None:
                    errors.append(f"Node '{node.id}' is a parent but has no comparison matrix set for its children.")
                else:
                    # Check if the matrix dimensions match the number of children.
                    if node.comparison_matrix.shape[0] != len(node.children):
                        errors.append(f"Matrix for Node '{node.id}' has dimensions {node.comparison_matrix.shape} "
                                      f"but it has {len(node.children)} children.")

                # Recurse down the tree
                for child in node.children:
                    _recursive_check(child)

        _recursive_check(model.root)
        return errors

    @staticmethod
    def validate_performance_scores(model: Hierarchy) -> List[str]:
        """
        Validates that every alternative has a performance score for every leaf node.

        Args:
            model: The Hierarchy instance to validate.

        Returns:
            A list of error strings describing missing performance scores.
        """
        errors = []
        if not model.alternatives:
            errors.append("Model has no alternatives to validate scores for.")
            return errors

        leaf_nodes = model.root.get_all_leaf_nodes()
        if not leaf_nodes:
            errors.append("Hierarchy has no leaf nodes to score against.")
            return errors

        for alt in model.alternatives:
            for leaf in leaf_nodes:
                if leaf.id not in alt.performance_scores:
                    errors.append(f"Alternative '{alt.name}' is missing a performance score for leaf node '{leaf.id}'.")

        return errors

    @staticmethod
    def run_all_validations(model: Hierarchy, tolerance: float = 1e-6) -> Dict[str, List[str]]:
        """
        Runs a complete suite of validations on the Hierarchy.

        Args:
            model: The Hierarchy instance to validate.
            tolerance: Tolerance for floating-point comparisons.

        Returns:
            A dictionary containing lists of errors for each validation category.
        """
        all_errors = {
            "hierarchy_completeness": [],
            "matrix_properties": [],
            "performance_scores": []
        }

        # 1. Validate hierarchy completeness (all necessary matrices are present)
        all_errors["hierarchy_completeness"] = Validation.validate_hierarchy_completeness(model)

        # 2. Validate properties of all existing matrices
        matrix_errors = []
        def _collect_matrices(node: 'Node'):
            if node.comparison_matrix is not None:
                errors = Validation.validate_matrix_properties(node.comparison_matrix, tolerance)
                if errors:
                    matrix_errors.extend([f"[Node: {node.id}] {e}" for e in errors])
            for child in node.children:
                _collect_matrices(child)

        _collect_matrices(model.root)
        all_errors["matrix_properties"] = matrix_errors

        # 3. Validate that all performance scores are set
        all_errors["performance_scores"] = Validation.validate_performance_scores(model)

        return all_errors

    @staticmethod
    def validate_matrix_dimensions(matrix: np.ndarray, expected_size: int | None = None) -> List[str]:
        """Validates that a matrix is a 2D square NumPy array of the expected size."""
        errors = []
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            errors.append("Input must be a 2D square NumPy array.")
            return errors # Stop further checks
        if expected_size is not None and matrix.shape[0] != expected_size:
            errors.append(f"Matrix has size {matrix.shape[0]}, but expected size {expected_size}.")
        return errors

    @staticmethod
    def validate_matrix_diagonal(matrix: np.ndarray, tolerance: float = 1e-9) -> List[str]:
        """Validates that diagonal elements are the multiplicative identity (e.g., 1)."""
        errors = []
        if matrix.shape[0] == 0: return errors # Handle empty matrix
        identity_one = matrix[0,0].multiplicative_identity()
        for i in range(matrix.shape[0]):
            if abs(matrix[i, i].centroid() - identity_one.centroid()) > tolerance:
                errors.append(f"Diagonal element at ({i},{i}) is not 1. Found: {matrix[i,i]}")
        return errors

    @staticmethod
    def validate_matrix_reciprocity(matrix: np.ndarray, tolerance: float = 1e-9) -> List[str]:
        """Validates the reciprocal property a_ji = 1/a_ij for the entire matrix."""
        errors = []
        n = matrix.shape[0]
        for i in range(n):
            for j in range(i, n):
                try:
                    inverse_val = matrix[j, i].inverse()
                    val = matrix[i, j]
                    # Compare the defuzzified centroids for a stable check
                    if abs(val.centroid() - inverse_val.centroid()) > tolerance:
                        errors.append(f"Reciprocity failed between ({i},{j}) and ({j},{i}). "
                                      f"Value: {val}, Inverse of counterpart: {inverse_val}")
                except Exception as e:
                    errors.append(f"Error checking reciprocity for ({i},{j}): {e}")
        return errors

    @staticmethod
    def run_all_matrix_validations(matrix: np.ndarray, expected_size: int | None = None, tolerance: float = 1e-9) -> Dict[str, List[str]]:
        """Runs a complete suite of validations on a single comparison matrix."""
        all_errors = {"dimensions": [], "diagonal": [], "reciprocity": []}
        all_errors["dimensions"] = Validation.validate_matrix_dimensions(matrix, expected_size)

        # Only run further checks if dimensions are valid
        if not all_errors["dimensions"]:
            all_errors["diagonal"] = Validation.validate_matrix_diagonal(matrix, tolerance)
            all_errors["reciprocity"] = Validation.validate_matrix_reciprocity(matrix, tolerance)
        return all_errors

    @staticmethod
    def check_model_completeness(model: Hierarchy) -> Dict[str, List[str]]:
        """
        Validates the completeness of the AHP model structure, checking for
        missing matrices and performance scores.
        """
        all_errors = {
            "hierarchy_completeness": [],
            "performance_scores": []
        }

        # 1. Validate hierarchy completeness (all necessary matrices are present)
        hierarchy_errors = []
        def _recursive_check(node: 'Node'):
            if not node.is_leaf:
                if node.comparison_matrix is None:
                    hierarchy_errors.append(f"Node '{node.id}' is a parent but has no comparison matrix set.")
                elif node.comparison_matrix.shape[0] != len(node.children):
                    hierarchy_errors.append(f"Matrix for Node '{node.id}' has dimensions {node.comparison_matrix.shape} but it has {len(node.children)} children.")
                for child in node.children:
                    _recursive_check(child)
        _recursive_check(model.root)
        all_errors["hierarchy_completeness"] = hierarchy_errors

        # 2. Validate that all performance scores are set
        score_errors = []
        if model.alternatives:
            leaf_nodes = model.root.get_all_leaf_nodes()
            if not leaf_nodes:
                score_errors.append("Hierarchy has no leaf nodes to score against.")
            else:
                for alt in model.alternatives:
                    for leaf in leaf_nodes:
                        if leaf.id not in alt.performance_scores:
                            score_errors.append(f"Alternative '{alt.name}' is missing a performance score for leaf node '{leaf.id}'.")
        all_errors["performance_scores"] = score_errors

        return all_errors
