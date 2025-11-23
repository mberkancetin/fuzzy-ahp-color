"""
===================================================================
Tests for the General AHP Workflow
===================================================================

This script contains integration tests that simulate a complete user workflow
from model creation to final ranking. It ensures that all modules in the
library work together as expected.
"""

import pytest
import numpy as np

# Import all necessary components from the library
from multiAHPy.model import Hierarchy, Node, Alternative
from multiAHPy.types import TFN, Crisp
from multiAHPy.matrix_builder import create_matrix_from_judgments
from multiAHPy.aggregation import aggregate_matrices
from multiAHPy.consistency import Consistency
from multiAHPy.validation import Validation

# --- Test Fixture for a complete, multi-level hierarchy problem ---

@pytest.fixture
def full_problem_setup():
    """Defines the structure and judgments for a complete AHP problem."""
    # Define the structure
    criteria = ["Price", "Performance"]
    sub_criteria_perf = ["CPU", "GPU"]
    alternatives = ["Laptop A", "Laptop B"]

    # Define the judgments from two experts
    # Expert 1 favors Price
    expert1_crit_judgments = {("Price", "Performance"): 3}
    expert1_perf_judgments = {("CPU", "GPU"): 2}

    # Expert 2 favors Performance
    expert2_crit_judgments = {("Price", "Performance"): 1/4}
    expert2_perf_judgments = {("CPU", "GPU"): 1/2}

    # Judgments for alternatives under each leaf criterion
    alt_vs_price = {("Laptop A", "Laptop B"): 1/5} # A is much cheaper
    alt_vs_cpu = {("Laptop A", "Laptop B"): 3}   # A has a better CPU
    alt_vs_gpu = {("Laptop A", "Laptop B"): 1/2} # B has a better GPU

    return {
        "criteria": criteria,
        "sub_criteria_perf": sub_criteria_perf,
        "alternatives": alternatives,
        "expert1": {"crit": expert1_crit_judgments, "perf": expert1_perf_judgments},
        "expert2": {"crit": expert2_crit_judgments, "perf": expert2_perf_judgments},
        "alt_judgments": {
            "Price": alt_vs_price,
            "CPU": alt_vs_cpu,
            "GPU": alt_vs_gpu
        }
    }

@pytest.fixture
def sample_tfn_model() -> Hierarchy:
    """
    Provides a basic but complete 2-level TFN model, now correctly set up
    for the AHP synthesis calculation.
    """
    # Define structure
    criteria = ["Cost", "Quality"]
    sub_criteria_cost = ["Purchase Price", "Maintenance"]
    alternatives = ["Option A", "Option B"]

    # Create model
    goal_node = Node("Goal", "Select Best Option")
    model = Hierarchy[TFN](goal_node, number_type=TFN)

    # Build hierarchy
    cost_node = Node("Cost")
    quality_node = Node("Quality")
    goal_node.add_child(cost_node)
    goal_node.add_child(quality_node)

    cost_node.add_child(Node("Purchase Price"))
    cost_node.add_child(Node("Maintenance"))

    # Add alternatives
    model.add_alternative("Option A")
    model.add_alternative("Option B")

    # Set criteria/sub-criteria comparison matrices
    crit_judgments = {("Cost", "Quality"): 2}
    crit_matrix = create_matrix_from_judgments(crit_judgments, criteria, TFN, fuzziness=1)
    model.set_comparison_matrix("Goal", crit_matrix)

    sub_crit_judgments = {("Purchase Price", "Maintenance"): 3}
    sub_crit_matrix = create_matrix_from_judgments(sub_crit_judgments, sub_criteria_cost, TFN, fuzziness=1)
    model.set_comparison_matrix("Cost", sub_crit_matrix)

    # Judgments for alternatives under each leaf criterion
    alt_vs_purchase_price = {("Option A", "Option B"): 1/4} # Option B is better on price
    alt_vs_maintenance = {("Option A", "Option B"): 1/2} # Option B is better on maintenance
    alt_vs_quality = {("Option A", "Option B"): 5}   # Option A is much better on quality

    # Create and set each matrix
    matrix_price = create_matrix_from_judgments(alt_vs_purchase_price, alternatives, TFN, fuzziness=1)
    model.set_alternative_matrix("Purchase Price", matrix_price)

    matrix_maint = create_matrix_from_judgments(alt_vs_maintenance, alternatives, TFN, fuzziness=1)
    model.set_alternative_matrix("Maintenance", matrix_maint)

    matrix_qual = create_matrix_from_judgments(alt_vs_quality, alternatives, TFN, fuzziness=1)
    model.set_alternative_matrix("Quality", matrix_qual)

    return model


def test_calculate_alternative_scores(sample_tfn_model):
    """Test that the calculate_alternative_scores method runs and populates scores."""
    # First, we must calculate weights
    sample_tfn_model.calculate_weights()

    # Before score calculation, overall_score is None
    alt_a = sample_tfn_model.get_alternative("Option A")
    assert alt_a.overall_score is None

    # Run score calculation (this is the method being tested)
    sample_tfn_model.rank_alternatives_by_comparison()

    # After calculation, scores should be populated
    assert isinstance(alt_a.overall_score, TFN)
    assert alt_a.overall_score.m > 0

def test_full_end_to_end_tfn_workflow(full_problem_setup):
    """
    Tests the entire Fuzzy AHP workflow from matrix creation to final ranking.
    This test verifies that all modules integrate correctly.
    """
    # Unpack test data
    data = full_problem_setup

    # 1. CREATE AND AGGREGATE MATRICES
    # ------------------------------------
    # Create individual matrices for each expert
    crit_matrix1 = create_matrix_from_judgments(data['expert1']['crit'], data['criteria'], TFN, fuzziness=1)
    crit_matrix2 = create_matrix_from_judgments(data['expert2']['crit'], data['criteria'], TFN, fuzziness=1)
    perf_matrix1 = create_matrix_from_judgments(data['expert1']['perf'], data['sub_criteria_perf'], TFN, fuzziness=1)
    perf_matrix2 = create_matrix_from_judgments(data['expert2']['perf'], data['sub_criteria_perf'], TFN, fuzziness=1)

    # Aggregate the experts' judgments
    group_crit_matrix = aggregate_matrices([crit_matrix1, crit_matrix2], method="geometric")
    group_perf_matrix = aggregate_matrices([perf_matrix1, perf_matrix2], method="geometric")

    assert group_crit_matrix.shape == (2, 2)
    assert isinstance(group_crit_matrix[0,1], TFN)

    # 2. BUILD THE HIERARCHY MODEL
    # -------------------------------
    goal = Node("Goal", "Choose Best Laptop")
    model = Hierarchy[TFN](goal, number_type=TFN)

    price_node = Node("Price")
    perf_node = Node("Performance")
    goal.add_child(price_node)
    goal.add_child(perf_node)

    perf_node.add_child(Node("CPU"))
    perf_node.add_child(Node("GPU"))

    for name in data['alternatives']:
        model.add_alternative(name)

    # 3. SET MATRICES IN THE MODEL
    # ------------------------------
    model.set_comparison_matrix("Goal", group_crit_matrix)
    model.set_comparison_matrix("Performance", group_perf_matrix)

    # Create and set alternative comparison matrices for each LEAF node
    # The leaf nodes are 'Price', 'CPU', and 'GPU'
    for leaf_id, judgments in data['alt_judgments'].items():
        alt_matrix = create_matrix_from_judgments(judgments, data['alternatives'], TFN, fuzziness=1)
        model.set_alternative_matrix(leaf_id, alt_matrix)

    # 4. VALIDATE AND CHECK CONSISTENCY
    # -----------------------------------
    # This checks that all necessary matrices are present
    completeness_errors = Validation.check_model_completeness(model)
    assert not completeness_errors['hierarchy_completeness'] # Should be no errors

    # Check consistency (we don't assert consistency, just that it runs)
    consistency_report = model.check_consistency()
    assert "Goal" in consistency_report
    assert "Performance" in consistency_report
    assert "Price" in consistency_report

    # 5. RUN CALCULATIONS AND VERIFY RESULTS
    # --------------------------------------
    model.calculate_weights()
    model.rank_alternatives_by_comparison()

    # Check that key nodes have valid weights
    cpu_node = model._find_node("CPU")
    assert isinstance(cpu_node.global_weight, TFN)
    assert cpu_node.global_weight.defuzzify() > 0
    assert cpu_node.global_weight.defuzzify() < 1

    # Check that alternatives have valid scores
    alt_a = model.get_alternative("Laptop A")
    assert isinstance(alt_a.overall_score, TFN)
    assert 0 < alt_a.overall_score.defuzzify() < 1

    # 6. GET FINAL RANKING
    # --------------------
    rankings = model.get_rankings()
    assert len(rankings) == 2
    assert rankings[0][1] >= rankings[1][1] # Check that it's sorted

    print(f"\nEnd-to-end test successful. Final ranking: {rankings}")
