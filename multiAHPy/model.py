from __future__ import annotations
import numpy as np
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any, Type, Generic, Literal
from .consistency import Consistency
from .weight_derivation import derive_weights
from .types import TFN, IFN, Number
from copy import deepcopy
import copy

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

    def add_child(self, child: Node[Number]):
        """
        Adds a child node and establishes the parent-child link.
        """
        child.parent = self
        self.children.append(child)
        if self.model: child._set_model_reference(self.model)

    def _set_model_reference(self, model: 'Hierarchy[Number]'):
        """Recursively sets the model reference for this node and all its children."""
        self.model = model
        for child in self.children: child._set_model_reference(model)

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
        return len(self.children) == 0

    def get_all_leaf_nodes(self) -> List[Node[Number]]:
        """
        Recursively finds all leaf nodes under this node.
        """
        if self.is_leaf: return [self]
        leaves = []
        for c in self.children: leaves.extend(c.get_all_leaf_nodes())
        return leaves

    def get_all_nodes(self) -> List[Node]:
        """Recursively finds all nodes (including self) under this node."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes

    def calculate_global_weights(self):
        """
        Recursively calculates the global weights for this node's children.
        Assumes its own global_weight has already been set.
        """
        # Recursive calculation
        # If using IFN, be careful with multiplication.
        # Often it is better to propagate crisp weights if fuzzy multiplication explodes uncertainty.
        if self.global_weight is None: return

        for child in self.children:
            if child.local_weight:
                child.global_weight = self.global_weight * child.local_weight
                child.calculate_global_weights()

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

    def __deepcopy__(self, memo):
        """
        Custom deepcopy implementation for the Alternative class.
        This creates a new Alternative but intentionally does NOT copy the
        cached results (`node_scores`, `overall_score`), ensuring that any
        new analysis starts with a clean slate.
        """
        # Create a new, blank instance of the class
        cls = self.__class__
        new_alt = cls.__new__(cls)
        memo[id(self)] = new_alt # Add to memo to prevent infinite loops

        # Copy the fundamental attributes, but not the results
        new_alt.name = self.name
        new_alt.description = self.description
        new_alt.performance_scores = copy.deepcopy(self.performance_scores, memo)

        # Intentionally create empty result caches for the new object
        new_alt.node_scores = {}
        new_alt.overall_score = None

        return new_alt

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

    def to_dataframe(self, defuzzify_method: str = 'centroid') -> 'pd.DataFrame':
        """
        Exports the hierarchy structure and weights to a pandas DataFrame.

        Args:
            defuzzify_method: The method to use for converting fuzzy weights to crisp values.

        Returns:
            A pandas DataFrame containing the node ID, description, local weight,
            global weight, and consistency metrics (if available).
        """
        _check_pandas_availability()

        data = []

        def _traverse(node: Node):
            local_w = node.local_weight.defuzzify(method=defuzzify_method) if node.local_weight else None
            global_w = node.global_weight.defuzzify(method=defuzzify_method) if node.global_weight else None

            consistency_report = getattr(self, 'consistency_report', {}).get(node.id, {})

            data.append({
                'node_id': node.id,
                'description': node.description,
                'parent_id': node.parent.id if node.parent else None,
                'is_leaf': node.is_leaf,
                'local_weight': local_w,
                'global_weight': global_w,
                'saaty_cr': consistency_report.get('saaty_cr'),
                'gci': consistency_report.get('gci'),
                'is_consistent': consistency_report.get('is_consistent')
            })

            for child in node.children:
                _traverse(child)

        _traverse(self.root)
        return pd.DataFrame(data)

    @classmethod
    def from_dataframe(cls, df: 'pd.DataFrame', number_type: Type[Number], root_id: str) -> 'Hierarchy':
        """
        Constructs a Hierarchy model from a pandas DataFrame containing node structure.

        The DataFrame must contain 'node_id' and 'parent_id' columns.

        Args:
            df: The DataFrame containing the hierarchy structure.
            number_type: The NumericType class to use for the model (e.g., TFN, Crisp).
            root_id: The ID of the root node in the DataFrame.

        Returns:
            A new Hierarchy instance.
        """
        _check_pandas_availability()

        if 'node_id' not in df.columns or 'parent_id' not in df.columns:
            raise ValueError("DataFrame must contain 'node_id' and 'parent_id' columns.")

        nodes: Dict[str, Node] = {}
        for _, row in df.iterrows():
            node_id = row['node_id']
            description = row.get('description')
            nodes[node_id] = Node(node_id, description)

        root_node = None
        for _, row in df.iterrows():
            node_id = row['node_id']
            parent_id = row['parent_id']

            if node_id == root_id:
                root_node = nodes[node_id]

            if parent_id and parent_id in nodes:
                nodes[parent_id].add_child(nodes[node_id])

        if root_node is None:
            raise ValueError(f"Root node with ID '{root_id}' not found in DataFrame.")

        return cls(root_node, number_type)

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
        start = start_node or self.root
        if start.id == node_id: return start
        for c in start.children:
            found = self._find_node(node_id, c)
            if found: return found
        return None

    def get_all_nodes(self):
        return self.root.get_all_nodes()

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

        parent_node.comparison_matrix = matrix

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
                        converted_matrix[i, j] = target_type.from_saaty(cell)
                    except Exception as e:
                        raise TypeError(f"Could not convert cell at ({i},{j}) with value '{cell}' to type '{target_type.__name__}'. Error: {e}")

        criterion_node.comparison_matrix = converted_matrix

    def calculate_weights(self, method: str = "geometric_mean"):
        """
        Calculates all local and global weights for the entire hierarchy.
        """
        # Recursive derivation
        def _calc(node):
            if not node.is_leaf and node.comparison_matrix is not None:
                # Use 'normalized_score' for IFN consistency checks by default to ensure positive weights
                c_method = 'normalized_score' if self.number_type.__name__ == 'IFN' else 'centroid'

                res = derive_weights(node.comparison_matrix, self.number_type, method, consistency_method=c_method)

                weights = res['weights']
                for i, child in enumerate(node.children):
                    child.local_weight = weights[i]

            for c in node.children: _calc(c)

        _calc(self.root)

        # Set root global weight
        self.root.global_weight = self.number_type.multiplicative_identity()
        self.root.calculate_global_weights()

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

    def _recalculate_global_weights(self):
        """
        Recursively recalculates all global weights in the hierarchy based on
        the currently set local weights.

        This method does NOT re-derive weights from matrices. It's used for
        sensitivity analysis where local weights are adjusted manually.
        """
        # Ensure the root is set up correctly
        if self.root.global_weight is None:
            self.root.global_weight = self.number_type.multiplicative_identity()
        # Start the propagation from the root node
        self._propagate_weights(self.root)

    def _propagate_weights(self, node: Node):
        # For each child, calculate its global weight and then recurse
        for child in node.children:
            if node.global_weight is None or child.local_weight is None:
                # This might happen if weights were never calculated at all
                raise RuntimeError(f"Cannot propagate weights: parent '{node.id}' or child '{child.id}' has a missing weight.")

            # The core logic: global_child = global_parent * local_child
            child.global_weight = node.global_weight * child.local_weight

            # Recurse down the tree
            self._propagate_weights(child)

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
            return self.number_type.from_normalized(priority_value)

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

        # For IFN, override default centroid with 'normalized_score' if not specified, to prevent negative weights
        if self.number_type.__name__ == 'IFN' and consistency_method == 'centroid':
            consistency_method = 'normalized_score'

        leaf_nodes = self.root.get_all_leaf_nodes()

        leaves = self.root.get_all_leaf_nodes()

        if not as_dict: return leaves

        raw_vals = []
        keys = []

        for leaf in leaves:
            keys.append(leaf.id)
            if defuzzify:
                val = leaf.global_weight.defuzzify(method=consistency_method)
                raw_vals.append(max(0, val)) # Clamp
            else:
                raw_vals.append(leaf.global_weight)

        if not defuzzify:
            return dict(zip(keys, raw_vals))

        # Normalize
        total = sum(raw_vals)
        if total > 0:
            norm_vals = [v / total for v in raw_vals]
        else:
            norm_vals = [1.0/len(raw_vals)] * len(raw_vals)

        return dict(zip(keys, norm_vals))

    def get_child_weights(
        self,
        parent_node_id: str,
        weight_type: Literal["local", "global"] = "local"
    ) -> Dict[str, float]:
        """
        Returns the final weights of the children of a specified parent node.

        Args:
            parent_node_id (str): The ID of the parent node whose children's
                                weights are to be retrieved.
            weight_type (str, optional): The type of weight to return.
                                        - 'local': (Default) The weights of the children
                                                    relative to their parent (should sum to 1.0).
                                        - 'global': The weights of the children relative
                                                    to the overall goal.

        Returns:
            A dictionary mapping { 'child_node_id': crisp_weight }.
        """
        parent_node = self._find_node(parent_node_id)
        if parent_node is None:
            raise ValueError(f"Parent node '{parent_node_id}' not found.")

        weights_dict = {}
        for child in parent_node.children:
            weight_to_get = None
            if weight_type == "local":
                weight_to_get = child.local_weight
            elif weight_type == "global":
                weight_to_get = child.global_weight
            else:
                raise ValueError("weight_type must be 'local' or 'global'.")

            if weight_to_get is None:
                raise RuntimeError(f"Weight of type '{weight_type}' has not been calculated for node '{child.id}'. Run `calculate_weights()` first.")

            # Use the default, safe defuzzification method
            weights_dict[child.id] = weight_to_get.defuzzify()

        # --- NORMALIZATION SAFEGUARD ---
        # The local weights of a set of siblings should always sum to 1.0.
        # We add a normalization step here to correct for any minor floating-point
        # artifacts from fuzzy arithmetic.
        if weight_type == "local":
            total_weight = sum(weights_dict.values())
            if abs(total_weight) > 1e-9:
                for child_id in weights_dict:
                    weights_dict[child_id] /= total_weight

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
        if not self.alternatives: return
        if self.number_type.__name__ == 'IFN':
            for alt in self.alternatives:
                alt.node_scores = {}
                alt.overall_score = self._calculate_performance_score_recursive_ifn(self.root, alt)
        else:
            for alt in self.alternatives:
                alt.node_scores = {}
                alt.overall_score = self._calculate_score_recursive_arithmetic(self.root, alt)
        print("Alternative performance scoring complete.")

    def score_alternatives_by_performance_ifwa(self):
        """
        Calculates final scores for alternatives based on pre-defined performance scores.
        This is the 'AHP as a Weighting Engine' or 'Scoring Model' workflow.

        This method requires that:
        1. Criteria weights have been calculated with `calculate_weights()`.
        2. Each alternative has a normalized performance score (0 to 1) set for EVERY
        leaf node using `alt.set_performance_score()`.
        """
        if not self.alternatives:
            print("Warning: No alternatives to score.")
            return

        if self.number_type.__name__ == 'IFN':

            leaf_nodes = self.root.get_all_leaf_nodes()

            for alt in self.alternatives:
                # Gather the performance scores and weights for this alternative
                perf_scores_as_ifn = []
                leaf_global_weights = []

                for leaf in leaf_nodes:
                    if leaf.id not in alt.performance_scores:
                        raise ValueError(f"Missing score for leaf '{leaf.id}' in alt '{alt.name}'.")

                    # Convert the crisp performance score to an IFN
                    perf_score_val = alt.get_performance_score(leaf.id)
                    perf_scores_as_ifn.append(self.number_type.from_normalized(perf_score_val))

                    # Get the crisp global weight of the leaf criterion
                    # We must use a safe, positive defuzzification method
                    leaf_global_weights.append(leaf.global_weight.defuzzify(method='centroid'))

                # Normalize the crisp weights to be sure they sum to 1
                weight_sum = sum(leaf_global_weights)
                normalized_weights = [w / weight_sum for w in leaf_global_weights]

                # --- Apply the IFWA formula ---
                # This is the same formula as in your `aggregate_priorities_ifwa`
                prod_1_minus_mu = np.prod([(1 - p.mu) ** w for p, w in zip(perf_scores_as_ifn, normalized_weights)])
                prod_nu = np.prod([p.nu ** w for p, w in zip(perf_scores_as_ifn, normalized_weights)])

                agg_mu = 1 - prod_1_minus_mu
                agg_nu = prod_nu

                alt.overall_score = self.number_type(agg_mu, agg_nu)

        else:
            for alt in self.alternatives:
                alt.overall_score = self._calculate_performance_score_recursive(self.root, alt)

        print("Alternative performance scoring complete.")

    def _calculate_performance_score_recursive_ifn(self, node: Node, alt: Alternative) -> Number:
        """
        Recursively calculates score using IFWA operator at each level.
        Populates node_scores for every visited node.
        """
        if node.id in alt.node_scores:
            return alt.node_scores[node.id]

        if node.is_leaf:
            raw_val = alt.get_performance_score(node.id)
            if raw_val is None:
                print(f"Warning: No score for {alt.name} on {node.id}, assuming 0.")
                raw_val = 0.0

            # Convert raw scalar (0-1) to IFN
            if hasattr(self.number_type, 'from_normalized'):
                score = self.number_type.from_normalized(raw_val)
            else:
                score = self.number_type(raw_val) # Fallback

            alt.node_scores[node.id] = score
            return score

        child_scores = []
        child_weights = []

        for child in node.children:
            c_score = self._calculate_performance_score_recursive_ifn(child, alt)
            child_scores.append(c_score)

            defuzz_meth = 'normalized_score' if self.number_type.__name__ == 'IFN' else 'centroid'
            w_val = child.local_weight.defuzzify(method=defuzz_meth)
            child_weights.append(max(0, w_val))

        total_w = sum(child_weights)
        if total_w > 1e-9:
            norm_weights = [w / total_w for w in child_weights]
        else:
            norm_weights = [1.0/len(child_weights)] * len(child_weights)

        # Calculate IFWA: (1 - prod((1-mu)^w), prod(nu^w))
        prod_mu_term = np.prod([(1 - s.mu)**w for s, w in zip(child_scores, norm_weights)])
        prod_nu_term = np.prod([s.nu**w for s, w in zip(child_scores, norm_weights)])

        agg_mu = 1 - prod_mu_term
        agg_nu = prod_nu_term

        agg_score = self.number_type(agg_mu, agg_nu)

        alt.node_scores[node.id] = agg_score
        return agg_score

    def _calculate_performance_score_recursive(self, node: Node[Number], alt: Alternative) -> Number:
        """
        (Helper for score_alternatives_by_performance)
        Recursively calculates the performance score of an alternative for a given node.
        """
        if node.id in alt.node_scores:
            return alt.node_scores[node.id]

        if node.is_leaf:
            score = self.number_type.from_normalized(alt.performance_scores[node.id])
            alt.node_scores[node.id] = score
            return score

        total_score = self.number_type.neutral_element()
        for child in node.children:
            child_score = self._calculate_performance_score_recursive(child, alt)
            total_score += child_score * child.local_weight

        alt.node_scores[node.id] = total_score
        return total_score

    def _calculate_score_recursive_arithmetic(self, node, alt):
        # Standard arithmetic recursion (SumProduct)
        if node.id in alt.node_scores: return alt.node_scores[node.id]

        if node.is_leaf:
            val = alt.get_performance_score(node.id) or 0.0
            score = self.number_type.from_normalized(val)
            alt.node_scores[node.id] = score
            return score

        # Sum(weight * child_score)
        total = self.number_type.neutral_element()
        for child in node.children:
            s = self._calculate_score_recursive_arithmetic(child, alt)
            total = total + (s * child.local_weight)

        alt.node_scores[node.id] = total
        return total

    def calculate_alternative_scores(self):
        """
        Universal wrapper to calculate final scores.
        Detects whether to use Ranking (Matrices) or Scoring (Performance Data).
        Required by visualization module.
        """
        if self.alternatives and any(a.performance_scores for a in self.alternatives):
            self.score_alternatives_by_performance()
            return
        leaves = self.root.get_all_leaf_nodes()
        if leaves and leaves[0].comparison_matrix is not None:
            self.rank_alternatives_by_comparison()
            return

        print("Warning: calculate_alternative_scores could not determine method (no data found).")

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

    def check_consistency(self, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Performs a comprehensive consistency check on all matrices in the model.

        This is a convenience method that delegates the calculation to the
        `Consistency.check_model_consistency` static method.

        Args:
            **kwargs: Can include 'consistency_method' and 'saaty_cr_threshold'.

        Returns:
            A dictionary with detailed consistency results for each matrix in the model.
        """
        if self.number_type.__name__ == 'IFN' and 'consistency_method' not in kwargs:
             kwargs['consistency_method'] = 'normalized_score'
        return Consistency.check_model_consistency(self, **kwargs)

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
        plot_sensitivity_analysis(self, parent_node_id, criterion_id, alt_name, figsize)
        return None

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
