from __future__ import annotations
from typing import Dict, Any, Literal, get_args

FuzzyNumberPreference = Literal["simple", "interval_certainty", "hesitation"]
AggregationGoal = Literal["average", "robust_to_outliers", "envelope", "consensus"]
WeightDerivationGoal = Literal["stable_and_simple", "true_fuzzy", "consistency_focused_crisp"]


class MethodSuggester:
    """
    Generates actionable configuration dictionaries for running Fuzzy AHP models.

    This class translates high-level research goals into the specific method names
    and parameters required by the multiAHPy library, based on the decision
    flowcharts and analysis in Liu, Y. et al. (2020), Expert Systems with Applications.

    Instead of just giving advice, it produces a dictionary that can be directly
    used in your analysis, similar to scikit-learn's get_params().

    Example:
        # User knows they need a simple model that is robust to outlier experts.
        recipe = MethodSuggester.get_model_recipe(
            fuzzy_number_preference="simple",
            aggregation_goal="robust_to_outliers"
        )

        # The output can be directly used:
        # >>> print(recipe)
        # {
        #     'number_type': <class 'multiAHPy.types.TFN'>,
        #     'aggregation_method': 'median',
        #     'weight_derivation_method': 'geometric_mean',
        #     'consistency_method': 'centroid',
        #     'number_type': 'TFN',
        # }

        model = Hierarchy(root, number_type=recipe['number_type'])
        agg_matrix = aggregate_matrices(matrices, method=recipe['aggregation_method'])
        model.calculate_weights(method=recipe['weight_derivation_method'])
    """
    @staticmethod
    def get_model_recipe(
        fuzzy_number_preference: FuzzyNumberPreference = "simple",
        aggregation_goal: AggregationGoal = "average",
        weight_derivation_goal: WeightDerivationGoal = "stable_and_simple",
        is_group_decision: bool = True
    ) -> Dict[str, Any]:
        """
        Generates a full "recipe" of methods and types for an AHP model.

        Args:
            fuzzy_number_preference: Describes the nature of expert uncertainty.
                - 'simple': Experts are reasonably confident, can provide a single peak value.
                            (Maps to TFN).
                - 'interval_certainty': Experts are certain about a range, not a single point.
                                         (Maps to TrFN).
                - 'hesitation': Experts are uncertain about membership vs. non-membership.
                                (Maps to IFN).
            aggregation_goal: The goal for combining multiple expert judgments.
                - 'average': Standard geometric/arithmetic mean. Best for homogenous groups.
                - 'robust_to_outliers': Use median to minimize impact of extreme outlier experts.
                - 'envelope': Capture the full range of all expert opinions (min-max).
                - 'consensus': Dynamically weight experts based on their agreement with the group.
            weight_derivation_goal: The desired property of the weight derivation process.
                - 'stable_and_simple': Use the most common, robust, and well-behaved method.
                                       (Maps to 'geometric_mean').
                - 'true_fuzzy': Preserve fuzziness throughout, extending eigenvector concepts.
                                (Maps to 'lambda_max').
                - 'consistency_focused_crisp': Prioritize finding a crisp weight vector that is
                                               most consistent with the fuzzy judgments.
                                               (Maps to 'fuzzy_programming').
            is_group_decision: Set to False if there is only one decision-maker.

        Returns:
            A dictionary containing the suggested `number_type`, `number_type_name`,
            `aggregation_method`, `weight_derivation_method`, and `consistency_method`.
        """
        from .types import TFN, TrFN, IFN

        if fuzzy_number_preference not in get_args(FuzzyNumberPreference):
            raise ValueError(f"Invalid fuzzy_number_preference. Choose from: {get_args(FuzzyNumberPreference)}")
        if aggregation_goal not in get_args(AggregationGoal):
            raise ValueError(f"Invalid aggregation_goal. Choose from: {get_args(AggregationGoal)}")
        if weight_derivation_goal not in get_args(WeightDerivationGoal):
            raise ValueError(f"Invalid weight_derivation_goal. Choose from: {get_args(WeightDerivationGoal)}")

        recipe = {}

        # --- Suggest Number Type ---
        if fuzzy_number_preference == "simple":
            recipe['number_type'] = TFN
        elif fuzzy_number_preference == "interval_certainty":
            recipe['number_type'] = TrFN
        elif fuzzy_number_preference == "hesitation":
            recipe['number_type'] = IFN
        else:
            recipe['number_type'] = TFN # Safe default

        # --- Suggest Aggregation Method ---
        if not is_group_decision:
            recipe['aggregation_method'] = None
        else:
            if recipe['number_type'] == IFN:
                recipe['aggregation_method'] = 'ifwa'
            else:
                if aggregation_goal == "average":
                    recipe['aggregation_method'] = 'geometric'
                elif aggregation_goal == "robust_to_outliers":
                    recipe['aggregation_method'] = 'median'
                elif aggregation_goal == "envelope":
                    recipe['aggregation_method'] = 'min_max'
                elif aggregation_goal == "consensus":
                    recipe['aggregation_method'] = 'median'

        # --- Suggest Weight Derivation Method ---
        if recipe['number_type'] == IFN:
             recipe['weight_derivation_method'] = 'geometric_mean'
             print("Guidance: For IFN, also consider using 'aggregate_priorities_ifwa' for group decisions.")
        else:
            if weight_derivation_goal == "stable_and_simple":
                recipe['weight_derivation_method'] = 'geometric_mean'
            elif weight_derivation_goal == "true_fuzzy":
                recipe['weight_derivation_method'] = 'lambda_max'
            elif weight_derivation_goal == "consistency_focused_crisp":
                recipe['weight_derivation_method'] = 'fuzzy_programming'

        # --- Suggest Defuzzification Method ---
        if recipe['number_type'] == IFN:
            recipe['consistency_method'] = 'score'
        else:
            recipe['consistency_method'] = 'centroid'

        recipe['number_type_name'] = recipe['number_type'].__name__
        return recipe

    @staticmethod
    def get_available_options() -> Dict[str, tuple]:
        """Returns all available choices for the recipe generator."""
        return {
            "fuzzy_number_preference": get_args(FuzzyNumberPreference),
            "aggregation_goal": get_args(AggregationGoal),
            "weight_derivation_goal": get_args(WeightDerivationGoal)
        }


suggester = MethodSuggester()
