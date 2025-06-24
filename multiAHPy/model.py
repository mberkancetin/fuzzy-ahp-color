from __future__ import annotations
import numpy as np
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any, Type, Generic
from multiAHPy.consistency import Consistency
from multiAHPy.defuzzification import Defuzzification, normalize_crisp_weights
from multiAHPy.weight_derivation import derive_weights
from multiAHPy.types import TFN, Number

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .types import TFN, TrFN, Crisp, GFN, NumericType, Number


class Node(Generic[Number]):
    """
    Represents a single node in a generic AHP hierarchy.
    A node can be a goal, criterion, sub-criterion, etc., allowing for arbitrary depth.
    """
    def __init__(self, node_id: str, description: Optional[str] = None):
        self.id = node_id
        self.description = description
        self.parent: Optional[Node[Number]] = None
        self.children: List[Node[Number]] = []
        self.comparison_matrix: Optional[np.ndarray] = None
        self.local_weight: Optional[Number] = None
        self.global_weight: Optional[Number] = None

    def __repr__(self) -> str:
        local_w_str = f"{self.local_weight.defuzzify():.3f}" if self.local_weight is not None else "N/A"
        global_w_str = f"{self.global_weight.defuzzify():.4f}" if self.global_weight is not None else "N/A"
        
        return (f"Node(id='{self.id}', local_weight={local_w_str}, "
                f"global_weight={global_w_str}, children={len(self.children)})")

    def add_child(self, child_node: Node[Number]):
        """
        Adds a child node and establishes the parent-child link.
        """
        child_node.parent = self
        self.children.append(child_node)

    @property
    def is_leaf(self) -> bool:
        """
        A node is a leaf if it has no children.
        """
        return not self.children

    def calculate_global_weights(self):
        """
        Recursively calculates the global weights for this node's children.
        Assumes its own global_weight has already been set.
        """
        # The logic is now focused only on the children.
        for child in self.children:
            if self.global_weight is None or child.local_weight is None:
                raise ValueError(f"Cannot calculate global weight for '{child.id}', as parent or local weight is missing.")

            # This will now be TFN * TFN or Crisp * Crisp, which works.
            child.global_weight = self.global_weight * child.local_weight

            # Recurse down the tree
            child.calculate_global_weights()


    def get_all_leaf_nodes(self) -> List[Node[Number]]:
        """
        Recursively finds all leaf nodes under this node.
        """
        if self.is_leaf:
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaf_nodes())
        return leaves


class Alternative(Generic[Number]):
    """
    Represents an alternative (e.g., a company) and stores its performance scores
    against the hierarchy's leaf nodes.
    """
    def __init__(self, name: str, description: Optional[str] = None):
        if not name:
            raise ValueError("Alternative name cannot be empty.")
        self.name = name
        self.description = description
        self.performance_scores: Dict[str, float] = {}
        self.node_scores: Dict[str, float] = {}
        self.overall_score: Optional[float] = None

    def __repr__(self):
        score_str = f"{self.overall_score:.4f}" if self.overall_score is not None else "N/A"
        return f"Alternative(name='{self.name}', overall_score={score_str})"

    def set_performance_score(self, leaf_node_id: str, score: float):
        """
        Sets the normalized performance score for a specific leaf node.
        """
        if not (0.0 <= score <= 1.0):
            print(f"Warning: Score for {leaf_node_id} is {score}, which is outside the typical normalized 0-1 range.")
        self.performance_scores[leaf_node_id] = score


class Hierarchy(Generic[Number]):
    """
    The main solver class that manages the hierarchy, weights, alternatives,
    and performs the final scoring calculations.
    """
    def __init__(self, root_node: Node[Number], number_type: Type[Number]):
        self.root = root_node
        self.alternatives: List[Alternative[Number]] = []
        self.number_type = number_type

    def add_alternative(self, alternative: Alternative):
        self.alternatives.append(alternative)

    def _find_node(self, node_id: str, start_node: Optional[Node[Number]] = None) -> Optional[Node[Number]]:
        """Helper to find a node by its ID anywhere in the tree."""
        if start_node is None:
            start_node = self.root
        if start_node.id == node_id:
            return start_node
        for child in start_node.children:
            found = self._find_node(node_id, child)
            if found:
                return found
        return None

    def set_comparison_matrix(self, parent_node_id: str, matrix: np.ndarray):
        """
        Sets the pairwise comparison matrix for the children of a given node.
        """
        parent_node = self._find_node(parent_node_id)
        if parent_node is None:
            raise ValueError(f"Node '{parent_node_id}' not found.")
        if len(parent_node.children) != matrix.shape[0]:
            raise ValueError(f"Matrix dimensions ({matrix.shape[0]}) do not match the number of children ({len(parent_node.children)}) for parent '{parent_node_id}'.")
        parent_node.comparison_matrix = matrix

    def calculate_weights(self, method: str = "extent_analysis"):
        """
        Calculates all local and global weights for the entire hierarchy.
        """
        print("\n--- Calculating Local Weights ---")
        self._calculate_local_weights_recursive(self.root, method)

        print("\n--- Calculating Global Weights ---")
        # Set the root's global weight to the correct type of "one"
        self.root.global_weight = self.number_type.multiplicative_identity()

        self.root.calculate_global_weights()
        print("Global weights calculation complete.")

    def _calculate_local_weights_recursive(self, node: Node[Number], method: str):
        """
        Helper for recursive local weight calculation.
        """
        if not node.is_leaf and node.comparison_matrix is not None:
            weights = derive_weights(node.comparison_matrix, self.number_type, method)
            if len(weights) != len(node.children):
                raise RuntimeError("Mismatch between number of derived weights and children.")
            for i, child_node in enumerate(node.children):
                child_node.local_weight = weights[i]
        for child in node.children:
            self._calculate_local_weights_recursive(child, method)

    def calculate_alternative_scores(self):
        """
        Calculates the scores for all alternatives at every level of the hierarchy.
        """
        print("\n--- Calculating Alternative Scores ---")
        if not self.alternatives:
            print("No alternatives to score.")
            return

        leaf_nodes = self.root.get_all_leaf_nodes()
        leaf_node_ids = {leaf.id for leaf in leaf_nodes}

        for alt in self.alternatives:
            # Validate that all necessary performance scores are set
            for leaf_id in leaf_node_ids:
                if leaf_id not in alt.performance_scores:
                    raise ValueError(f"Missing performance score for leaf node '{leaf_id}' in alternative '{alt.name}'.")

            # Kick off the recursive, bottom-up calculation for each alternative
            alt.overall_score = self._calculate_node_score_for_alt(self.root, alt)

        print("Alternative scores calculation complete.")

    def _calculate_node_score_for_alt(self, node: Node[Number], alt: Alternative) -> float:
        """
        Recursively calculates the performance score of an alternative for a given node.
        """
        if node.id in alt.node_scores:
            return alt.node_scores[node.id]

        if node.is_leaf:
            score = alt.performance_scores[node.id]
            alt.node_scores[node.id] = score
            return score

        # Recursive step: score is the weighted sum of its children's scores.
        total_score = 0.0
        for child in node.children:
            child_score = self._calculate_node_score_for_alt(child, alt)
            total_score += child_score * child.local_weight

        alt.node_scores[node.id] = total_score
        return total_score

    def get_rankings(self, defuzzify_method: str = 'centroid', **kwargs) -> List[Tuple[str, float]]:
        """
        Returns a sorted list of alternatives based on their overall scores.
        If scores are fuzzy, it defuzzifies them using the specified method
        before sorting to produce a crisp ranking.

        Args:
            defuzzify_method (str, optional): The method to use for defuzzification.
                                              Defaults to 'centroid'.
            **kwargs: Additional arguments to pass to the defuzzification function
                      (e.g., alpha for alpha_cut).

        Returns:
            List[Tuple[str, float]]: A sorted list of (alternative_name, crisp_score).
        """
        if not self.alternatives or self.alternatives[0].overall_score is None:
            raise RuntimeError("Scores not calculated. Run `calculate_alternative_scores()` first.")

        crisp_rankings = []
        for alt in self.alternatives:
            score = alt.overall_score
            crisp_score = score.defuzzify(method=defuzzify_method, **kwargs)
            crisp_rankings.append((alt.name, crisp_score))

        crisp_rankings.sort(key=lambda x: x[1], reverse=True)
        return crisp_rankings

    def check_consistency(self, defuzzify_method: str = 'centroid', threshold: float = 0.1) -> Dict[str, Dict[str, Any]]:
        """
        Performs a comprehensive consistency check on all matrices in the model
        by delegating to the Consistency class.

        Args:
            defuzzify_method (str, optional): The defuzzification method to use. Defaults to 'centroid'.
            threshold (float, optional): The acceptable CR threshold. Defaults to 0.1.

        Returns:
            A dictionary with detailed consistency results for each matrix in the model.
        """
        print("\n--- Checking All Matrix Consistencies ---")
        # Simply call the static method from our new Consistency class
        return Consistency.check_model_consistency(self, defuzzify_method, threshold)

    def display(self):
        """
        Displays a horizontal, column-based HTML representation of the AHP hierarchy.
        """
        from .visualization import display_tree_hierarchy
        display_tree_hierarchy(self)

    def summary(self, alternative_name: str) -> None:
        """Prints a detailed text summary of the model for a given alternative."""
        from .visualization import format_model_summary
        print(format_model_summary(self, alternative_name))

    def plot_weights(self, parent_node_id: str, figsize=(10, 6)):
        """Plots the local weights of the children of a given node."""
        from .visualization import plot_weights
        return plot_weights(self, parent_node_id, figsize)

    def plot_rankings(self, figsize=(10, 6)):
        """Plots the final rankings of the alternatives."""
        from .visualization import plot_final_rankings
        return plot_final_rankings(self, figsize)

    def plot_sensitivity(self, parent_node_id: str, criterion_id: str, alt_name: str, figsize=(12, 7)):
        """Plots a sensitivity analysis for a given criterion and alternative."""
        from .visualization import plot_sensitivity_analysis
        return plot_sensitivity_analysis(self, parent_node_id, criterion_id, alt_name, figsize)

    def full_report(self, filename: str | None = None, **kwargs) -> str:
        """
        Generates and prints or saves a comprehensive text report of all
        matrices, weights, and consistency metrics in the model.

        Args:
            filename (optional): Path to save the report as a .txt file.
            **kwargs: a_method, c_method
        Returns:
            The complete report as a string.
        """
        from .visualization import generate_full_report
        # Allow user to specify methods, otherwise use defaults
        d_method = kwargs.get('derivation_method', 'geometric_mean')
        c_method = kwargs.get('consistency_method', 'centroid')

        return generate_full_report(self, derivation_method=d_method,
                                    consistency_method=c_method, filename=filename)

    def export_report(self, filename: str, output_format: str = 'excel', **kwargs):
        """
        Exports a detailed analysis of all matrices to a specified format.

        Args:
            filename: For 'excel'/'csv', the path/base name.
                      For 'gsheet', the desired name of the new spreadsheet.
            output_format (str, optional): 'excel', 'csv', or 'gsheet'. Defaults to 'excel'.
            **kwargs: Can include 'derivation_method' and 'consistency_method'.
        """
        from .visualization import export_full_report
        d_method = kwargs.get('derivation_method', 'geometric_mean')
        c_method = kwargs.get('consistency_method', 'centroid')
        
        export_full_report(
            model=self,
            filename=filename,
            output_format=output_format,
            derivation_method=d_method,
            consistency_method=c_method
        )
