# tests/test_pipeline.py

import pytest
import numpy as np
import pandas as pd
import multiAHPy as ahp
from multiAHPy.model import Node
from multiAHPy.pipeline import Workflow
from multiAHPy.types import TFN, IFN
from multiAHPy.matrix_builder import create_matrix_from_list

# ==============================================================================
# 1. FIXTURES
# ==============================================================================

@pytest.fixture
def hierarchy_structure():
    """A simple 2-criterion hierarchy."""
    goal = Node("Goal")
    goal.add_child(Node("Price"))
    goal.add_child(Node("Quality"))
    return goal

@pytest.fixture
def alternatives_list():
    """A simple list of three alternatives."""
    return ["Car A", "Car B", "Car C"]

@pytest.fixture
def tfn_expert_matrices(alternatives_list):
    """
    Generates TFN-based judgments from two hypothetical experts.
    Expert 1: Prefers Price over Quality, and A over B/C on Price.
    Expert 2: Prefers Quality over Price, and B over A/C on Quality.
    """
    num_alts = len(alternatives_list)
    num_alt_judgments = num_alts * (num_alts - 1) // 2

    # Judgments for criteria (Price vs Quality)
    e1_crit = create_matrix_from_list([3], TFN) # E1: Price is more important
    e2_crit = create_matrix_from_list([1/4], TFN) # E2: Quality is more important

    # Judgments for alternatives under 'Price'
    e1_price = create_matrix_from_list([5, 3, 1/2], TFN) # E1: A >> B, A > C
    e2_price = create_matrix_from_list([4, 2, 1/2], TFN) # E2: A >> B, A > C

    # Judgments for alternatives under 'Quality'
    e1_quality = create_matrix_from_list([1/4, 1/2, 2], TFN) # E1: B > A, C > A
    e2_quality = create_matrix_from_list([1/5, 1/3, 2], TFN) # E2: B >> A, C > A

    return {
        "Goal": [e1_crit, e2_crit],
        "Price": [e1_price, e2_price],
        "Quality": [e1_quality, e2_quality]
    }

@pytest.fixture
def performance_score_data():
    """Provides normalized performance scores for the 'scoring' workflow."""
    return {
        "Car A": {"Price": 0.9, "Quality": 0.5},
        "Car B": {"Price": 0.4, "Quality": 0.9},
        "Car C": {"Price": 0.6, "Quality": 0.7}
    }

# ==============================================================================
# 2. PIPELINE WORKFLOW TESTS
# ==============================================================================

def test_pipeline_ranking_with_aggregate_judgments(hierarchy_structure, alternatives_list, tfn_expert_matrices):
    """
    TEST 1/4: Ranking workflow with AIJ (Aggregate Judgments).
    This is the most common "classic" group AHP workflow.
    """
    recipe = ahp.suggester.get_model_recipe(fuzzy_number_preference="simple", aggregation_goal="average")

    pipeline = Workflow(
        root_node=hierarchy_structure,
        workflow_type="ranking",
        group_strategy="aggregate_judgments",
        recipe=recipe,
        alternatives=alternatives_list
    )

    # Run the entire pipeline with all matrices
    pipeline.fit_weights(expert_matrices={"Goal": tfn_expert_matrices["Goal"]})
    pipeline.rank(alternative_matrices={
        "Price": tfn_expert_matrices["Price"],
        "Quality": tfn_expert_matrices["Quality"]
    })

    # --- Assertions ---
    assert pipeline.rankings != None
    assert len(pipeline.rankings) == 3
    # Based on the data, Car B should be favored due to Expert 2's strong preference for Quality
    assert pipeline.rankings[0][0] in ["Car B", "Car A"] # Rank can be close
    assert pipeline.consistency_report != None
    assert "aggregated_model" in pipeline.consistency_report
    assert "Goal" in pipeline.consistency_report["aggregated_model"]
    assert "is_consistent" in pipeline.consistency_report["aggregated_model"]["Goal"]


def test_pipeline_scoring_with_aggregate_judgments(hierarchy_structure, alternatives_list, tfn_expert_matrices, performance_score_data):
    """
    TEST 2/4: Scoring workflow with AIJ (Aggregate Judgments).
    This is the "weighting engine" use case with group consensus on weights.
    """
    recipe = ahp.suggester.get_model_recipe(fuzzy_number_preference="simple", aggregation_goal="average")

    pipeline = Workflow(
        root_node=hierarchy_structure,
        workflow_type="scoring",
        group_strategy="aggregate_judgments",
        recipe=recipe,
        alternatives=alternatives_list
    )

    # Step 1: Fit weights using only the criteria matrices
    pipeline.fit_weights(expert_matrices={"Goal": tfn_expert_matrices["Goal"]})

    # Step 2: Rank/score using the performance data
    pipeline.rank(performance_scores=performance_score_data)

    # --- Assertions ---
    assert pipeline.criteria_weights != None
    assert "Price" in pipeline.criteria_weights
    assert pipeline.rankings != None
    assert len(pipeline.rankings) == 3
    # With AIJ, weights are averaged, Price and Quality are roughly equal.
    # Car A has the best profile, so it should win.
    assert pipeline.rankings[0][0] == "Car A"

def test_pipeline_ranking_with_aggregate_priorities(hierarchy_structure, alternatives_list, tfn_expert_matrices):
    """
    TEST 3/4: Ranking workflow with AIP (Aggregate Priorities).

    This test verifies the complex workflow where the entire AHP model is solved
    for each expert individually, and their final ranking vectors are aggregated.
    """
    recipe = ahp.suggester.get_model_recipe(fuzzy_number_preference="simple")

    pipeline = Workflow(
        root_node=hierarchy_structure,
        workflow_type="ranking",
        group_strategy="aggregate_priorities",
        recipe=recipe,
        alternatives=alternatives_list
    )

    # --- Execute the pipeline ---
    # The .rank() method for AIP in a 'ranking' workflow requires the full set of
    # expert matrices for both criteria and alternatives.
    # Note: For AIP, there is no separate "fit_weights" step. The entire
    # process is done within the .rank() method.
    pipeline.fit_weights(expert_matrices={"Goal": tfn_expert_matrices["Goal"]})
    pipeline.rank(alternative_matrices=tfn_expert_matrices)

    # --- Assertions ---

    # 1. Check that the final rankings were produced and have the correct format
    assert pipeline.rankings != None, "Rankings should be calculated"
    assert len(pipeline.rankings) == len(alternatives_list), "Should have a ranking for each alternative"
    assert isinstance(pipeline.rankings[0], tuple), "Rankings should be a list of tuples"
    assert isinstance(pipeline.rankings[0][0], str), "First element of ranking tuple should be a string name"
    assert isinstance(pipeline.rankings[0][1], float), "Second element of ranking tuple should be a float score"

    # 2. Check the ranking order based on our analysis of the expert preferences
    assert pipeline.rankings != None
    assert len(pipeline.rankings) == 3

    # FIX: Assert the correct calculated order
    # Car B should be first, Car A second, Car C third.
    assert pipeline.rankings[0][0] == 'Car B'
    assert pipeline.rankings[1][0] == 'Car A'
    assert pipeline.rankings[2][0] == 'Car C'

    # You can also assert that the scores for A and B are very close
    assert abs(pipeline.rankings[0][1] - pipeline.rankings[1][1]) < 0.01


    # 3. Check that the consistency report was generated correctly for the AIP strategy
    assert pipeline.consistency_report != None, "Consistency report should be generated for AIP"
    # For AIP, we expect a pandas DataFrame summarizing each expert
    assert isinstance(pipeline.consistency_report, dict), "AIP consistency report should be a DataFrame"
    assert "expert_1" in pipeline.consistency_report.keys()
    assert "Goal" in pipeline.consistency_report["expert_1"].keys()
    assert "is_consistent" in pipeline.consistency_report["expert_1"]["Goal"].keys()

    # Check that there are entries for both experts
    assert "matrix_size" in pipeline.consistency_report["expert_2"]["Goal"].keys()
    assert "matrix_size" in pipeline.consistency_report["expert_2"]["Goal"].keys()

def test_pipeline_scoring_with_aggregate_priorities(hierarchy_structure, alternatives_list, tfn_expert_matrices, performance_score_data):
    """
    TEST 4/4: Scoring workflow with AIP (Aggregate Priorities).
    This workflow averages the final criteria weights of each expert.
    """
    recipe = ahp.suggester.get_model_recipe(fuzzy_number_preference="simple")

    pipeline = Workflow(
        root_node=hierarchy_structure,
        workflow_type="scoring",
        group_strategy="aggregate_priorities",
        recipe=recipe,
        alternatives=alternatives_list
    )

    # Step 1: Fit weights using AIP
    pipeline.fit_weights(expert_matrices={"Goal": tfn_expert_matrices["Goal"]})

    # Step 2: Score using performance data
    pipeline.rank(performance_scores=performance_score_data)

    # --- Assertions ---
    assert pipeline.criteria_weights != None
    assert pipeline.consistency_report.__class__ == dict # AIP should produce a dict report
    assert "matrix_size" in pipeline.consistency_report["expert_1"]["Goal"].keys()

    # Let's check the weights. E1 prefers Price (weight > 0.5), E2 prefers Quality (Price weight < 0.5).
    # The final aggregated weight for Price should be around the average.
    assert 0.4 < pipeline.criteria_weights["Price"] < 0.6

    assert pipeline.rankings != None
    # Again, Car A has the best profile and should win.
    print(pipeline)
    assert pipeline.rankings[0][0] == "Car A"


def test_pipeline_error_on_missing_data(hierarchy_structure, alternatives_list):
    """
    Tests that the pipeline raises appropriate errors if data is missing.
    """
    recipe = ahp.suggester.get_model_recipe()
    pipeline = Workflow(
        root_node=hierarchy_structure,
        workflow_type="scoring",
        group_strategy="aggregate_judgments",
        recipe=recipe,
        alternatives=alternatives_list
    )

    # Try to rank without fitting weights first
    with pytest.raises(RuntimeError, match="Weights have not been fitted yet"):
        pipeline.rank(performance_scores=performance_score_data)

    # Fit weights, then try to rank with missing performance scores
    pipeline.fit_weights(expert_matrices={"Goal": [create_matrix_from_list([2], TFN)]})
    import re
    with pytest.raises(ValueError, match=re.escape("The 'scoring' workflow requires the 'performance_scores' argument.")):
        pipeline.rank()


def test_pipeline_scoring_with_aggregate_priorities2(hierarchy_structure, alternatives_list, tfn_expert_matrices, performance_score_data):
    """
    TEST 4/4: Scoring workflow with AIP (Aggregate Priorities).
    """
    recipe = ahp.suggester.get_model_recipe(fuzzy_number_preference="simple")

    pipeline = Workflow(
        root_node=hierarchy_structure,
        workflow_type="scoring",
        group_strategy="aggregate_priorities", # <-- Specify AIP
        recipe=recipe,
        alternatives=alternatives_list
    )

    # Step 1: Fit weights using the AIP strategy
    # We only need to provide matrices for nodes with sub-criteria.
    pipeline.fit_weights(expert_matrices={"Goal": tfn_expert_matrices["Goal"]})

    # Assert that weights were calculated
    assert pipeline.criteria_weights != None
    assert "Price" in pipeline.criteria_weights

    # Step 2: Score alternatives using the fitted weights and performance data
    pipeline.score(performance_scores=performance_score_data)

    # Assert that rankings were produced
    assert pipeline.rankings != None
    assert len(pipeline.rankings) == 3
    assert pipeline.rankings[0][0] == "Car A" # Based on the data, Car A should win
