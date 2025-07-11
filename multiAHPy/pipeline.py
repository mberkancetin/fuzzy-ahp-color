from __future__ import annotations
from typing import List, Dict, Any, Literal, Tuple
from copy import deepcopy
import json
import numpy as np
from .model import Hierarchy, Node
from .types import NumericType, TFN, Crisp, IFN, TrFN, GFN, IT2TrFN
from .aggregation import aggregate_matrices, aggregate_priorities
from .matrix_builder import create_matrix_from_list


TYPE_MAP = {
    "Crisp": Crisp, "TFN": TFN, "IFN": IFN, "TrFN": TrFN, "GFN": GFN, "IT2TrFN": IT2TrFN
}

WorkflowType = Literal["ranking", "scoring"]
GroupStrategy = Literal["aggregate_judgments", "aggregate_priorities"]


class Workflow:
    """
    A pipeline for executing a complete AHP analysis from data to results.

    This class encapsulates the entire sequence of operations for a specific AHP
    workflow, ensuring that steps are not missed and that the process is
    reproducible. It's inspired by scikit-learn's Pipeline object.

    Workflow Steps:
    1. Initialize the workflow with a hierarchy structure and a full method recipe.
    2. Provide the necessary data (expert matrices or performance scores).
    3. Run the pipeline to get the final results.

    """

    def __init__(
        self,
        root_node: Node,
        workflow_type: WorkflowType,
        recipe: Dict[str, Any],
        alternatives: List[str],
        group_strategy: GroupStrategy = "aggregate_judgments",
        completion_method: str | None = None
    ):
        """
        Initializes the AHP Workflow Pipeline.

        Args:
            root_node: The root node of the AHP hierarchy structure.
            workflow_type: The type of analysis to perform.
                           - 'ranking': Classic AHP to rank alternatives via pairwise comparison.
                           - 'scoring': AHP as a weighting engine to score alternatives on performance data.
            recipe: A dictionary of methods, typically generated by `ahp.suggester`.
                    Must contain 'number_type', 'aggregation_method', 'weight_derivation_method'.
            alternatives: A list of the names of the alternatives.
        """
        if workflow_type not in ["ranking", "scoring"]:
            raise ValueError("workflow_type must be either 'ranking' or 'scoring'.")

        self.workflow_type = workflow_type
        self.group_strategy = group_strategy
        self.recipe = recipe
        self._root_node_template = root_node
        self.alternatives = alternatives
        self.rankings: List[Tuple[str, float]] | None = None
        self.consistency_report: Dict[str, Any] | None = None
        self.criteria_weights: Dict[str, float] | None = None
        self.completion_method = completion_method
        self.num_missing_in_report: Dict[str, int] = {}
        self.model = self._create_model_instance()

    def display(self):
        """
        Displays a horizontal, column-based HTML representation of the AHP hierarchy.
        """
        from .visualization import display_tree_hierarchy
        display_tree_hierarchy(self.model)

    @classmethod
    def from_json(cls, json_string: str) -> 'Workflow':
        """
        Parses a JSON string and executes the defined AHP workflow.

        Args:
            json_string: A string containing the full workflow definition in the
                         specified JSON schema.

        Returns:
            The fitted pipeline object with results.
        """
        data = json.loads(json_string)

        config = data['workflow_config']
        recipe = config['recipe']

        number_type_str = recipe.get("number_type", "Crisp")
        number_type = TYPE_MAP.get(number_type_str)
        if not number_type:
            raise ValueError(f"Unsupported number_type in JSON recipe: '{number_type_str}'")
        recipe['number_type'] = number_type

        root_node = Node.from_dict(data['hierarchy'])
        alternatives = [alt['name'] for alt in data['alternatives']]

        pipeline = cls(
            root_node=root_node,
            workflow_type=config['workflow_type'],
            recipe=recipe,
            alternatives=alternatives,
            group_strategy=config.get('group_strategy', 'aggregate_judgments'),
            completion_method=config.get('completion_method')
        )

        expert_matrices: Dict[str, List[np.ndarray]] = {}
        expert_weights: List[float] = []

        if 'expert_judgments' in data:
            all_matrix_nodes = set()
            for expert_data in data['expert_judgments']:
                all_matrix_nodes.update(expert_data['matrices'].keys())

            for node_id in all_matrix_nodes:
                expert_matrices[node_id] = []

            for expert_data in data['expert_judgments']:
                expert_weights.append(expert_data.get('weight'))
                for node_id, matrix_data in expert_data['matrices'].items():
                    matrix = create_matrix_from_list(matrix_data[0], number_type)
                    expert_matrices[node_id].append(matrix)

        if all(w is not None for w in expert_weights):
            weight_sum = sum(expert_weights)
            if weight_sum > 0:
                expert_weights = [w / weight_sum for w in expert_weights]
        else:
            expert_weights = None

        pipeline.run(
            expert_matrices=expert_matrices,
            performance_scores=data.get('performance_scores'),
            expert_weights=expert_weights
        )

        return pipeline

    def _create_model_instance(self) -> Hierarchy:
        """
        Creates a fresh, deep-copied instance of the Hierarchy model.

        This ensures that each expert calculation in the 'aggregate_priorities'
        strategy is completely isolated and does not modify a shared model object.
        It uses the template structure and recipe defined when the pipeline
        was initialized.
        """
        model_instance = Hierarchy(
            root_node=deepcopy(self._root_node_template),
            number_type=self.recipe['number_type']
        )

        for alt_name in self.alternatives:
            model_instance.add_alternative(alt_name)

        return model_instance

    def fit_weights(self, expert_matrices: Dict[str, List[np.ndarray]], expert_weights: List[float] | None = None):
        """
        Calculates the final criteria weights and checks consistency based on the chosen group strategy.

        This method populates the pipeline with `self.criteria_weights` and `self.consistency_report`.
        """
        print(f"\n--- Fitting Weights (Strategy: {self.group_strategy}) ---")
        self.consistency_report = {}

        if self.group_strategy == "aggregate_judgments":
            # STRATEGY 1: Aggregate Judgments First (AIJ)
            aggregated_matrices = {}
            agg_method = self.recipe.get('aggregation_method', 'geometric')

            for node_id, matrices in expert_matrices.items():
                if not self.model._find_node(node_id).is_leaf:
                    agg_matrix = aggregate_matrices(matrices, method=agg_method, expert_weights=expert_weights) \
                                if len(matrices) > 1 else matrices[0]
                    aggregated_matrices[node_id] = agg_matrix
                    self.model.set_comparison_matrix(node_id, agg_matrix)

            self.model.calculate_weights(method=self.recipe['weight_derivation_method'])

            print("  - Checking consistency of the aggregated model...")
            self.consistency_report['aggregated_model'] = self.model.check_consistency()

        elif self.group_strategy == "aggregate_priorities":
            # STRATEGY 2: Aggregate Priorities (AIP)

            num_experts = len(list(expert_matrices.values())[0])
            print(f"  - Checking consistency for {num_experts} individual experts...")

            for i in range(num_experts):
                expert_name = f"expert_{i+1}"
                expert_model = Hierarchy(self.model.root, number_type=self.recipe['number_type'])

                for node_id, matrices in expert_matrices.items():
                    if not expert_model._find_node(node_id).is_leaf:
                        expert_model.set_comparison_matrix(node_id, matrices[i])

                self.consistency_report[expert_name] = expert_model.check_consistency()

            criteria_matrices_per_node = {
                node_id: matrices for node_id, matrices in expert_matrices.items()
                if not self.model._find_node(node_id).is_leaf
            }
            final_crisp_weights = self._calculate_aip_weights(criteria_matrices_per_node, expert_weights)
            self._set_final_weights_on_model(final_crisp_weights)

        self.criteria_weights = self.model.get_criteria_weights()
        print("--- Weight fitting complete. ---")

        return self

    def _fit_weights_aij(self, expert_matrices, expert_weights):
        """Fits weights using Aggregation of Individual Judgments."""
        aggregated_matrices = self._aggregate_expert_matrices(expert_matrices, expert_weights)

        for node_id, matrix in aggregated_matrices.items():
            self.model.set_comparison_matrix(node_id, matrix)

        self.model.calculate_weights(method=self.recipe['weight_derivation_method'])
        self.consistency_report = self.model.check_consistency()

    def _fit_weights_aip(self, expert_matrices: Dict[str, List[np.ndarray]], expert_weights: List[float] | None = None):
        """
        Fits weights using Aggregation of Individual Priorities (AIP).
        """
        num_experts = len(list(expert_matrices.values())[0])
        individual_criteria_weights = []
        for i in range(num_experts):
            expert_model = self._create_model_instance()
            for node_id, matrices in expert_matrices.items():
                node = expert_model._find_node(node_id)
                if node and not node.is_leaf:
                    expert_model.set_comparison_matrix(node_id, matrices[i])
            expert_model.calculate_weights(method=self.recipe['weight_derivation_method'])
            crisp_weights_vector = list(expert_model.get_criteria_weights(as_dict=True).values())
            individual_criteria_weights.append(crisp_weights_vector)

        print("  - Aggregating individual criteria weight vectors...")
        weights_matrix = np.array(individual_criteria_weights)

        final_group_weights_vector = np.average(weights_matrix, axis=0, weights=expert_weights)
        final_group_weights_vector /= np.sum(final_group_weights_vector)

        print("  - Checking consistency for each expert's judgments...")
        from .consistency import Consistency
        template_model = self._create_model_instance()
        criteria_matrices = {nid: m for nid, m in expert_matrices.items() if not template_model._find_node(nid).is_leaf}
        defuzz_method_for_consistency = self.recipe.get('consistency_method', 'centroid')

        self.consistency_report = Consistency.check_group_consistency(
            model=template_model,
            expert_matrices=criteria_matrices,
            consistency_method=defuzz_method_for_consistency
        )
        leaf_nodes = self._create_model_instance().root.get_all_leaf_nodes()
        leaf_ids = [node.id for node in leaf_nodes]
        final_weights_dict = dict(zip(leaf_ids, final_group_weights_vector))
        self.criteria_weights = dict(zip(leaf_ids, final_group_weights_vector))

        return final_weights_dict

    def _score_by_performance(self, performance_scores):
        if self.group_strategy == 'aggregate_priorities':
            self._score_by_performance_aip(performance_scores)
        else:
            self._score_by_performance_aij(performance_scores)

    def _score_by_performance_aip(self, performance_scores: Dict[str, Dict[str, float]]):
        """
        (Private Helper) Scores alternatives using pre-aggregated AIP weights.
        This is a direct application of weights to performance data.
        """
        print("  - Applying aggregated priority (AIP) weights to performance scores...")

        for alt_obj in self.model.alternatives:
            alt_name = alt_obj.name

            if alt_name not in performance_scores:
                raise ValueError(f"Missing performance data for alternative '{alt_name}'.")

            alt_perf_data = performance_scores[alt_name]
            overall_score = 0.0

            for leaf_id, weight in self.criteria_weights.items():
                try:
                    # Score = weight of criterion * performance on that criterion
                    performance_value = alt_perf_data[leaf_id]
                    overall_score += weight * performance_value
                except KeyError:
                    raise KeyError(f"Missing performance score for alternative '{alt_name}' on criterion '{leaf_id}'.")

            alt_obj.overall_score = self.model.number_type.from_crisp(overall_score)

    def _score_by_performance_aij(self, performance_scores: Dict[str, Dict[str, float]]):
        """
        (Private Helper) Scores alternatives for the AIJ workflow.
        This delegates to the model's internal recursive calculation.
        """
        print("  - Applying aggregated judgment (AIJ) weights to performance scores...")
        for alt_name, scores in performance_scores.items():
            alt_obj = self.model.get_alternative(alt_name)
            for leaf_id, score in scores.items():
                alt_obj.set_performance_score(leaf_id, score)

        self.model.score_alternatives_by_performance()

    def _rank_by_comparison(self, alternative_matrices, expert_weights=None):
        """Ranks alternatives using pairwise comparison matrices."""
        aggregated_alt_matrices = self._aggregate_expert_matrices(alternative_matrices, expert_weights)

        for leaf_id, matrix in aggregated_alt_matrices.items():
            self.model.set_alternative_matrix(leaf_id, matrix)

        self.model.rank_alternatives_by_comparison(derivation_method=self.recipe['weight_derivation_method'])

    def _rank_by_comparison_aip(self, expert_matrices: Dict[str, List[np.ndarray]], expert_weights: List[float] | None = None):
        """
        (Private Helper) Ranks alternatives using the AIP strategy.
        This involves solving the full AHP model for each expert and aggregating
        the final ranking vectors.
        """
        print("  - Solving full AHP model for each expert (AIP strategy)...")
        num_experts = len(list(expert_matrices.values())[0])

        individual_ranking_vectors = []

        for i in range(num_experts):
            print(f"    - Solving for Expert {i+1}...")

            expert_model = self._create_model_instance()

            for node_id, matrices in expert_matrices.items():
                node = expert_model._find_node(node_id)
                if not node: continue
                if node.is_leaf:
                    expert_model.set_alternative_matrix(node_id, matrices[i])
                else:
                    expert_model.set_comparison_matrix(node_id, matrices[i])

            expert_model.calculate_weights(method=self.recipe['weight_derivation_method'])
            expert_model.rank_alternatives_by_comparison(derivation_method=self.recipe['weight_derivation_method'])

            expert_ranks = expert_model.get_rankings(consistency_method=self.recipe.get('consistency_method', 'centroid'))
            score_dict = dict(expert_ranks)
            ordered_scores = [score_dict[alt_name] for alt_name in self.alternatives]
            individual_ranking_vectors.append(ordered_scores)

        print("  - Aggregating individual final rankings...")
        rankings_matrix = np.array(individual_ranking_vectors)
        final_group_scores = np.average(rankings_matrix, axis=0, weights=expert_weights)

        for i, alt_obj in enumerate(self.model.alternatives):
            alt_obj.overall_score = self.model.number_type.from_crisp(final_group_scores[i])

        # Also set the consistency report (e.g., from the first expert as a sample)
        # A full implementation would run this for all experts.
        # ... (logic to generate group consistency table) ...

    def _aggregate_expert_matrices(self, expert_matrices, expert_weights):
        """Helper to aggregate a dictionary of expert matrices."""
        aggregated_data = {}
        aggregation_method = self.recipe.get('aggregation_method', 'geometric')
        for node_id, matrices in expert_matrices.items():
            if len(matrices) > 1:
                print(f"  - Aggregating {len(matrices)} matrices for node '{node_id}'...")
                agg_matrix = aggregate_matrices(matrices, method=aggregation_method, expert_weights=expert_weights, number_type=self.model.number_type)
                aggregated_data[node_id] = agg_matrix
            elif matrices:
                aggregated_data[node_id] = matrices[0]
        return aggregated_data

    def _calculate_aip_weights(
        self,
        criteria_matrices_per_node: Dict[str, List[np.ndarray]],
        expert_weights: List[float] | None
    ) -> Dict[str, np.ndarray]:
        """Helper to manage the complexity of the Aggregate Priorities workflow."""
        final_weights = {}

        for node_id, matrices in criteria_matrices_per_node.items():
            aggregated_vector = aggregate_priorities(
                matrices,
                method=self.recipe['weight_derivation_method'],
                expert_weights=expert_weights
            )
            final_weights[node_id] = aggregated_vector

        return final_weights

    def _set_final_weights_on_model(self, final_crisp_weights: Dict[str, np.ndarray]):
        """
        Manually sets the local and global weights on the main model.
        This is necessary after the AIP workflow.
        """
        number_type = self.recipe['number_type']

        def recursive_setter(node: Node, parent_global_weight: NumericType):
            node.global_weight = parent_global_weight * node.local_weight

            if node.id in final_crisp_weights:
                child_weights = final_crisp_weights[node.id]
                for i, child in enumerate(node.children):
                    child.local_weight = number_type.from_crisp(child_weights[i])

            for child in node.children:
                if not child.is_leaf:
                    recursive_setter(child, node.global_weight)

        root = self.model.root
        root.local_weight = number_type.multiplicative_identity()
        root.global_weight = number_type.multiplicative_identity()

        if root.id in final_crisp_weights:
            child_weights = final_crisp_weights[root.id]
            for i, child in enumerate(root.children):
                child.local_weight = number_type.from_crisp(child_weights[i])

        for child in root.children:
            recursive_setter(child, root.global_weight)

    def score(self, performance_scores: Dict[str, Dict[str, float]] | None = None) -> 'Workflow':
        """
        Runs the final scoring step of the 'scoring' workflow.
        Requires `fit_weights` to have been called first.
        """
        if self.workflow_type != "scoring":
            raise TypeError("The .score() method is only for the 'scoring' workflow.")
        if self.criteria_weights is None:
            raise RuntimeError("Must call .fit_weights() before .score()")

        print("\n--- Scoring alternatives based on performance data ---")

        if performance_scores is not None:
            print("  - Loading performance scores from provided dictionary...")
            for alt_name, scores in performance_scores.items():
                try:
                    alt_obj = self.model.get_alternative(alt_name)
                    for leaf_id, score_val in scores.items():
                        alt_obj.set_performance_score(leaf_id, score_val)
                except ValueError:
                    print(f"Warning: Alternative '{alt_name}' from performance_scores not found in model. Skipping.")
        else:
            print("  - Using pre-loaded performance scores from Alternative objects.")

        self.model.score_alternatives_by_performance()
        self.rankings = self.model.get_rankings()

        print("--- Scoring complete. ---")
        return self

    def run(
        self,
        expert_matrices: Dict[str, List[np.ndarray]],
        performance_scores: Dict[str, Dict[str, float]] | None = None,
        expert_weights: List[float] | None = None
    ) -> 'Workflow':
        """
        Executes the entire AHP pipeline with the provided data in one call.

        This is a convenience method that calls the appropriate sequence of
        `fit_weights`, `score`, or `rank` methods based on the workflow's
        configuration.

        Args:
            expert_matrices: A dictionary mapping a node ID to a list of expert
                             comparison matrices. Required for both workflows.
            performance_scores: A dictionary mapping an alternative name to its
                                performance scores. Required for 'scoring' workflow.
            expert_weights: Optional list of weights for the experts.

        Returns:
            The fitted pipeline object, with results in its attributes.
        """
        criteria_matrices_for_fit = {
            node_id: matrices for node_id, matrices in expert_matrices.items()
            if not self.model._find_node(node_id).is_leaf
        }
        self.fit_weights(criteria_matrices_for_fit, expert_weights)

        if self.workflow_type == 'ranking':
            alternative_matrices_for_rank = {
                node_id: matrices for node_id, matrices in expert_matrices.items()
                if self.model._find_node(node_id).is_leaf
            }
            self.rank(alternative_matrices_for_rank, expert_weights)

        elif self.workflow_type == 'scoring':
            if performance_scores is None:
                raise ValueError("The 'scoring' workflow requires the 'performance_scores' argument.")
            self.score(performance_scores)

        return self

    def rank(self, alternative_matrices: Dict[str, List[np.ndarray]], expert_weights: List[float] | None = None) -> 'Workflow':
        """Runs the final ranking step of the 'ranking' workflow."""
        if self.workflow_type != "ranking":
            raise TypeError("The .rank() method is only for the 'ranking' workflow.")
        if self.criteria_weights is None:
            raise RuntimeError("Must call .fit_weights() before .rank()")

        print("\n--- Ranking alternatives using pairwise comparison matrices ---")

        agg_method = self.recipe.get('aggregation_method', 'geometric')

        for leaf_id, matrices in alternative_matrices.items():
            agg_matrix = aggregate_matrices(matrices, method=agg_method, expert_weights=expert_weights) \
                         if len(matrices) > 1 else matrices[0]
            self.model.set_alternative_matrix(leaf_id, agg_matrix)

        self.model.rank_alternatives_by_comparison(derivation_method=self.recipe['weight_derivation_method'])
        self.rankings = self.model.get_rankings()

        print("--- Ranking complete. ---")
        return self
