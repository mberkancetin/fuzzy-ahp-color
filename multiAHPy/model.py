from __future__ import annotations
import numpy as np
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any, Type, Generic
from .consistency import Consistency
from .weight_derivation import derive_weights
from .types import TFN, Number
from copy import deepcopy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .types import TFN, TrFN, Crisp, GFN, NumericType, Number

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

def _check_pandas_availability():
    """Helper function to raise an error if pandas is not installed."""
    if not _PANDAS_AVAILABLE:
        raise ImportError("Excel/CSV export functionality requires the 'pandas' and 'openpyxl' libraries. "
                        "Please install them using: pip install pandas openpyxl")


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
        self.model: Optional['Hierarchy[Number]'] = None
        self.alternative_priorities: Dict[str, float] = {}

    def __repr__(self) -> str:
        local_w_str = f"{self.local_weight.defuzzify():.3f}" if self.local_weight else "N/A"
        global_w_str = f"{self.global_weight.defuzzify():.4f}" if self.global_weight else "N/A"
        return (f"Node(id='{self.id}', local_weight={local_w_str}, "
                f"global_weight={global_w_str}, children={len(self.children)})")

    def add_child(self, child_node: Node[Number]):
        """
        Adds a child node and establishes the parent-child link.
        """
        child_node.parent = self
        self.children.append(child_node)
        if self.model:
            child_node._set_model_reference(self.model)

    def _set_model_reference(self, model: 'Hierarchy[Number]'):
        """Recursively sets the model reference for this node and all its children."""
        self.model = model
        for child in self.children:
            child._set_model_reference(model)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Node and its children to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "children": [child.to_dict() for child in self.children]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Creates a Node instance and its descendants from a dictionary."""
        node = cls(node_id=data['id'], description=data.get('description'))
        if 'children' in data:
            for child_data in data['children']:
                child_node = cls.from_dict(child_data)
                node.add_child(child_node)
        return node

    def get_model(self) -> 'Hierarchy[Number]':
            """Returns the Hierarchy model this node belongs to."""
            if self.model is None:
                raise RuntimeError("This node is not part of a Hierarchy model.")
            return self.model

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
        for child in self.children:
            if self.global_weight is None or child.local_weight is None:
                raise ValueError(f"Cannot calculate global weight for '{child.id}', as parent or local weight is missing.")

            child.global_weight = self.global_weight * child.local_weight
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

    def judgments_to_table(
        self,
        group_matrices: List[np.ndarray] | None = None,
        source_names: List[str] | None = None,
        consistency_method: str | None = None
    ) -> 'pd.DataFrame':
        """
        Exports the comparison judgments for this node's children to a pandas DataFrame.

        - If `group_matrices` is provided, it will format the data in the
          flattened "Pairs" style, showing each source's judgment.
        - If `group_matrices` is NOT provided, it will format the node's own
          `comparison_matrix` into a classic n x n table.

        Args:
            group_matrices (optional): A list of matrices from multiple experts/sources.
                                        If None, uses the node's own comparison_matrix.
            source_names (optional): A list of names for each expert/source.
            consistency_method (optional): If provided, defuzzifies fuzzy numbers.

        Returns:
            A pandas DataFrame of the judgments.
        """
        from .visualization import format_group_judgments_as_table, format_matrix_as_table
        _check_pandas_availability()

        if self.is_leaf:
            if not hasattr(self, 'model'):
                raise RuntimeError("Node must be part of a Hierarchy to find alternative names.")
            item_names = [alt.name for alt in self.model.alternatives]
        else:
            item_names = [child.id for child in self.children]

        if group_matrices:
            return format_group_judgments_as_table(
                matrices=group_matrices,
                item_names=item_names,
                source_names=source_names,
                consistency_method=consistency_method
            )
        else:
            if self.comparison_matrix is None:
                raise ValueError(f"Node '{self.id}' has no comparison_matrix set.")
            return format_matrix_as_table(
                matrix=self.comparison_matrix,
                item_names=item_names,
                consistency_method=consistency_method
            )


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

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Alternative to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alternative':
        """Creates an Alternative instance from a dictionary."""
        return cls(name=data['name'], description=data.get('description'))

    def set_performance_score(self, leaf_node_id: str, score: float):
        """
        Sets the normalized performance score for a specific leaf node.
        """
        if not (0.0 <= score <= 1.0):
            print(f"Warning: Score for {leaf_node_id} is {score}, which is outside the typical normalized 0-1 range.")
        self.performance_scores[leaf_node_id] = score

    def get_performance_score(self, leaf_node_id: str) -> float | None:
        """Retrieves the raw performance score for a given leaf node."""
        return self.performance_scores.get(leaf_node_id)

    def get_score_at_node(self, node_id: str) -> Number | None:
        """
        Retrieves the calculated, synthesized score for this alternative at a
        specific node in the hierarchy.

        This represents the total weighted contribution of the branch under that node
        to this alternative's final score.
        """
        return self.node_scores.get(node_id)

    def clear_results(self):
        """Resets all calculated scores for this alternative."""
        self.node_scores.clear()
        self.overall_score = None


class Hierarchy(Generic[Number]):
    """
    The main solver class that manages the hierarchy, weights, alternatives,
    and performs the final scoring calculations.
    """
    def __init__(self, root_node: Node[Number], number_type: Type[Number]):
        self.root = root_node
        self.alternatives: List[Alternative[Number]] = []
        self.number_type = number_type
        self.last_used_derivation_method: str | None = None
        self.last_used_aggregation_method: str | None = None
        self.last_used_ranking_defuzz_method: str | None = None
        self.root._set_model_reference(self)

    @classmethod
    def from_json(cls, json_string: str, number_type: Type[Number]) -> 'Hierarchy':
        """
        Constructs a complete Hierarchy model from a JSON string.

        The JSON should define the hierarchy structure and alternatives.
        """
        data = json.loads(json_string)

        if "hierarchy" not in data or "alternatives" not in data:
            raise ValueError("JSON string must contain 'hierarchy' and 'alternatives' keys.")

        root_node = Node.from_dict(data['hierarchy'])
        model = cls(root_node, number_type)

        for alt_data in data['alternatives']:
            model.add_alternative(alt_data['name'], alt_data.get('description'))

        return model

    def add_alternative(self, name: str, description: Optional[str] = None):
        """
        Adds a new alternative to the model.

        Args:
            name: The unique name of the alternative.
            description: An optional description.

        Raises:
            ValueError: If an alternative with the same name already exists.
        """
        if any(alt.name == name for alt in self.alternatives):
            raise ValueError(f"Alternative '{name}' already exists in the model.")

        new_alt = Alternative(name, description)
        self.alternatives.append(new_alt)

    def get_alternative(self, name: str) -> Alternative[Number]:
        """
        Retrieves an alternative object by its name.

        Args:
            name: The name of the alternative to find.

        Returns:
            The Alternative object with the matching name.

        Raises:
            ValueError: If no alternative with the given name is found.
        """
        for alt in self.alternatives:
            if alt.name == name:
                return alt
        # If the loop finishes without finding the alternative, raise an error.
        raise ValueError(f"Alternative '{name}' not found in the model.")

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
        Sets the pairwise comparison matrix for the children of a given parent node.

        This method will automatically convert the elements of the provided matrix
        to the Hierarchy's specified `number_type` if they are not already.
        """
        parent_node = self._find_node(parent_node_id)
        if parent_node is None:
            raise ValueError(f"Node '{parent_node_id}' not found.")
        if parent_node.is_leaf:
            raise ValueError(f"Cannot set a sub-criteria comparison matrix for a leaf node ('{parent_node_id}').")
        if len(parent_node.children) != matrix.shape[0]:
            raise ValueError(f"Matrix dimensions ({matrix.shape[0]}) do not match the number of children ({len(parent_node.children)}) for parent '{parent_node_id}'.")

        n = matrix.shape[0]
        converted_matrix = np.empty((n, n), dtype=object)

        target_type = self.number_type

        for i in range(n):
            for j in range(n):
                cell = matrix[i, j]
                if isinstance(cell, target_type):
                    converted_matrix[i, j] = cell
                else:
                    try:
                        converted_matrix[i, j] = target_type.from_crisp(cell)
                    except Exception as e:
                        raise TypeError(f"Could not convert cell at ({i},{j}) with value '{cell}' to type '{target_type.__name__}'. Error: {e}")

        parent_node.comparison_matrix = converted_matrix

    def set_alternative_matrix(self, criterion_id: str, matrix: np.ndarray):
        """
        Sets the pairwise comparison matrix for the **alternatives** with respect
        to a specific LEAF criterion.

        Args:
            criterion_id: The ID of the leaf node criterion for this comparison.
            matrix: The n x n comparison matrix, where n is the number of alternatives.
        """
        criterion_node = self._find_node(criterion_id)
        if criterion_node is None:
            raise ValueError(f"Criterion node '{criterion_id}' not found.")
        if not criterion_node.is_leaf:
            raise ValueError(f"Can only set an alternative matrix for a leaf node criterion. '{criterion_id}' has children.")
        if len(self.alternatives) != matrix.shape[0]:
            raise ValueError(f"Alternative matrix dimensions ({matrix.shape[0]}) do not match the number of alternatives ({len(self.alternatives)}).")

        n = matrix.shape[0]
        converted_matrix = np.empty((n, n), dtype=object)

        target_type = self.number_type

        for i in range(n):
            for j in range(n):
                cell = matrix[i, j]
                if isinstance(cell, target_type):
                    converted_matrix[i, j] = cell
                else:
                    try:
                        converted_matrix[i, j] = target_type.from_crisp(cell)
                    except Exception as e:
                        raise TypeError(f"Could not convert cell at ({i},{j}) with value '{cell}' to type '{target_type.__name__}'. Error: {e}")

        criterion_node.comparison_matrix = converted_matrix

    def calculate_weights(self, method: str = "geometric_mean"):
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
        self.last_used_derivation_method = method

    def _calculate_local_weights_recursive(self, node: Node[Number], method: str):
        """
        Helper for recursive local weight calculation.
        """
        if not node.is_leaf and node.comparison_matrix is not None:
            results_dict = derive_weights(node.comparison_matrix, self.number_type, method)
            weights = results_dict['weights']
            if len(weights) != len(node.children):
                raise RuntimeError("Mismatch between number of derived weights and children.")
            for i, child_node in enumerate(node.children):
                child_node.local_weight = weights[i]
        for child in node.children:
            self._calculate_local_weights_recursive(child, method)

    def rank_alternatives_by_comparison(self, derivation_method: str = 'geometric_mean'):
        """
        Calculates final scores and ranks alternatives using the 'classic' AHP
        workflow, where alternatives are pairwise compared under each leaf criterion.

        This method is the final synthesis step. It requires that:
        1. Criteria weights have been calculated with `calculate_weights()`.
        2. An alternative comparison matrix has been set for EVERY leaf node
        using `set_alternative_matrix()`.

        Args:
            derivation_method (str, optional): The method to use for deriving the
                                            priorities of alternatives from their
                                            comparison matrices.
                                            Defaults to 'geometric_mean'.
        """
        from .weight_derivation import derive_weights

        print("\n--- Ranking Alternatives via Pairwise Comparison ---")

        if not self.alternatives:
            print("Warning: No alternatives in the model to rank.")
            return

        leaf_nodes = self.root.get_all_leaf_nodes()
        if not leaf_nodes:
            raise ValueError("Hierarchy has no leaf nodes to calculate scores against.")

        alt_local_priorities = {}

        for leaf in leaf_nodes:
            if leaf.comparison_matrix is None:
                raise ValueError(f"Leaf node '{leaf.id}' is missing its alternative comparison matrix, which is required for this method.")

            results = derive_weights(leaf.comparison_matrix, self.number_type, method=derivation_method)
            alt_local_priorities[leaf.id] = results['crisp_weights']

        for i, alt in enumerate(self.alternatives):
            alt.overall_score = self._calculate_score_recursive(self.root, i, alt_local_priorities)
            alt.node_scores[self.root.id] = alt.overall_score # Store the final score at the root

        print("Alternative ranking calculation complete.")

    def _calculate_score_recursive(
        self,
        node: Node[Number],
        alternative_index: int,
        alt_local_priorities: Dict[str, np.ndarray]
    ) -> Number:
        """
        (Helper for rank_alternatives_by_comparison)
        Recursively calculates the score for a single alternative from a specific node downwards.
        """
        if node.is_leaf:
            priority_value = alt_local_priorities[node.id][alternative_index]
            return self.number_type.from_crisp(priority_value)

        total_score = self.number_type.neutral_element()
        for child in node.children:
            child_score_contribution = self._calculate_score_recursive(child, alternative_index, alt_local_priorities)
            total_score += child_score_contribution * child.local_weight

        alt = self.alternatives[alternative_index]
        alt.node_scores[node.id] = total_score

        return total_score

    def get_criteria_weights(
        self, as_dict: bool = True,
        defuzzify: bool = True,
        consistency_method: str = "centroid"
        ) -> Dict[str, float] | List[Node]:
        """
        Returns the final global weights of all leaf-node criteria.

        This is the primary output when using AHP as a weighting engine. It should
        be called after `calculate_weights()`.

        Args:
            as_dict (bool, optional): If True (default), returns a dictionary of
                                      { 'node_id': crisp_weight }.
                                      If False, returns the list of leaf Node objects.
            defuzzify (bool, optional): If True (default) and as_dict is True,
                                        the weights in the dictionary will be
                                        crisp float values. If False, they will
                                        be the fuzzy number objects.

        Returns:
            A dictionary of weights or a list of Node objects.
        """
        if self.root.global_weight is None:
            raise RuntimeError("Weights have not been calculated yet. Call `calculate_weights()` first.")

        leaf_nodes = self.root.get_all_leaf_nodes()

        if not as_dict:
            return leaf_nodes

        defuzz_method = consistency_method
        weights_dict = {}
        for leaf in leaf_nodes:
            if leaf.global_weight is None:
                raise RuntimeError(f"Global weight for leaf node '{leaf.id}' is missing.")

            if defuzzify:
                weights_dict[leaf.id] = leaf.global_weight.defuzzify(method=defuzz_method)
            else:
                weights_dict[leaf.id] = leaf.global_weight

        return weights_dict

    def score_alternatives_by_performance(self):
        """
        Calculates final scores for alternatives based on pre-defined performance scores.
        This is the 'AHP as a Weighting Engine' or 'Scoring Model' workflow.

        This method requires that:
        1. Criteria weights have been calculated with `calculate_weights()`.
        2. Each alternative has a normalized performance score (0 to 1) set for EVERY
        leaf node using `alt.set_performance_score()`.
        """
        print("\n--- Scoring Alternatives via Performance Data ---")
        if not self.alternatives:
            print("Warning: No alternatives to score.")
            return

        leaf_nodes = self.root.get_all_leaf_nodes()
        leaf_node_ids = {leaf.id for leaf in leaf_nodes}

        for alt in self.alternatives:
            for leaf_id in leaf_node_ids:
                if leaf_id not in alt.performance_scores:
                    raise ValueError(f"Missing performance score for leaf node '{leaf_id}' in alternative '{alt.name}'. Use `alt.set_performance_score()`.")

            alt.overall_score = self._calculate_performance_score_recursive(self.root, alt)

        print("Alternative performance scoring complete.")

    def _calculate_performance_score_recursive(self, node: Node[Number], alt: Alternative) -> Number:
        """
        (Helper for score_alternatives_by_performance)
        Recursively calculates the performance score of an alternative for a given node.
        """
        if node.id in alt.node_scores:
            return alt.node_scores[node.id]

        if node.is_leaf:
            score = self.number_type.from_crisp(alt.performance_scores[node.id])
            alt.node_scores[node.id] = score
            return score

        total_score = self.number_type.neutral_element()
        for child in node.children:
            child_score = self._calculate_performance_score_recursive(child, alt)
            total_score += child_score * child.local_weight

        alt.node_scores[node.id] = total_score
        return total_score

    def get_rankings(self, consistency_method: str = 'centroid', **kwargs) -> List[Tuple[str, float]]:
        """
        Returns a sorted list of alternatives based on their overall scores.
        If scores are fuzzy, it defuzzifies them using the specified method
        before sorting to produce a crisp ranking.

        Args:
            consistency_method (str, optional): The method to use for defuzzification.
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
            crisp_score = score.defuzzify(method=consistency_method, **kwargs)
            crisp_rankings.append((alt.name, crisp_score))

        crisp_rankings.sort(key=lambda x: x[1], reverse=True)
        self.last_used_ranking_defuzz_method = consistency_method

        return crisp_rankings

    def check_consistency(
        self,
        consistency_method: str = 'centroid',
        saaty_cr_threshold: float | None = None,
        num_missing_map: Dict[str, int] | None = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Performs a comprehensive consistency check on all matrices in the model,
        calculating all registered consistency metrics.

        Args:
            consistency_method: The defuzzification method used for CR and GCI.
            saaty_cr_threshold: The acceptable threshold for Saaty's CR. If None,
                                uses the value from ahp_config.
            num_missing_map: A dict mapping node IDs to the number of missing
                             pairs in their matrix. Used to select the correct
                             generalized Random Index.
        Returns:
            A dictionary where keys are node IDs and values are a detailed
            dictionary of all calculated consistency metrics.
        """
        from .consistency import Consistency, CONSISTENCY_METHODS
        from .config import configure_parameters

        if num_missing_map is None:
            num_missing_map = {}

        final_cr_threshold = saaty_cr_threshold if saaty_cr_threshold is not None else configure_parameters.DEFAULT_SAATY_CR_THRESHOLD

        full_report = {}

        def _traverse_and_check(node: Node):
            if node.comparison_matrix is not None:
                matrix = node.comparison_matrix
                n = matrix.shape[0]
                number_type = type(matrix[0, 0])
                num_missing = num_missing_map.get(node.id, 0)
                node_results = {"matrix_size": n, "num_missing_pairs": num_missing}

                for name, func in CONSISTENCY_METHODS.items():
                    try:
                        node_results[name] = func(
                            matrix=matrix,
                            number_type=number_type,
                            consistency_method=consistency_method,
                            num_missing_pairs=num_missing
                        )
                    except Exception as e:
                        node_results[name] = f"Error: {e}"

                gci_threshold = Consistency._get_gci_threshold(n)
                cr_val = node_results.get("saaty_cr", float('inf'))
                gci_val = node_results.get("gci", float('inf'))

                is_cr_ok = isinstance(cr_val, (int, float)) and cr_val <= final_cr_threshold
                is_gci_ok = isinstance(gci_val, (int, float)) and gci_val <= gci_threshold

                node_results["is_consistent"] = is_cr_ok and is_gci_ok
                node_results["saaty_cr_threshold"] = final_cr_threshold
                node_results["gci_threshold"] = gci_threshold

                full_report[node.id] = node_results

            for child in node.children:
                _traverse_and_check(child)

        _traverse_and_check(self.root)

        return full_report

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

    def export_report(
        self,
        target: str,
        output_format: str = 'excel',
        spreadsheet_id: str | None = None,
        derivation_method: str = 'geometric_mean',
        consistency_method: str = 'centroid'
    ):
        """
        Exports a detailed analysis report.

        Args:
            target: For 'excel'/'csv', the path. For 'gsheet' (create mode), the new sheet's name.
            output_format: 'excel', 'csv', or 'gsheet'.
            spreadsheet_id (optional): If provided for 'gsheet', the script will open and
                                       update this existing spreadsheet by its ID/key.
            derivation_method: str = 'geometric_mean', # comparison matrix weight derivation methods
                The method to use, one of:
                    Crisp: "geometric_mean", "eigenvector"
                    TFN: "geometric_mean", "extent_analysis", "llsm"
                    TrFN: "geometric_mean", "extent_analysis", "llsm"
                    GFN: "geometric_mean"
            consistency_method: str = 'centroid' # defuzzification methods
                The method to use, one of:
                    Crisp: "centroid"
                    TFN: "centroid", "graded_mean", "alpha_cut", "weighted_average", "pessimistic", "optimistic"
                    TrFN: "centroid", "average"
                    GFN: "centroid", "pessimistic_99_percent"
        """
        from .visualization import export_full_report
        export_full_report(
            model=self,
            target=target,
            output_format=output_format,
            spreadsheet_id=spreadsheet_id,
            derivation_method=derivation_method,
            consistency_method=consistency_method
        )
