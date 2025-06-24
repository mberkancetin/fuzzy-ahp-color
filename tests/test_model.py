"""
===================================================================
Tests for the Model Module
===================================================================

This script contains unit tests for the core data structures of the library:
Hierarchy (AHPModel), Node, and Alternative.
"""

import pytest
import numpy as np

# Import the code to be tested
from multiAHPy.model import Hierarchy, Node, Alternative
from multiAHPy.types import TFN, Crisp
from multiAHPy.matrix_builder import create_matrix_from_judgments

# --- Fixture for a basic, fully-defined model ---

@pytest.fixture
def sample_tfn_model() -> Hierarchy:
    """Provides a basic but complete 2-level TFN model for testing."""
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

    # Set matrices
    # Goal vs Criteria
    crit_judgments = {("Cost", "Quality"): 2}
    crit_matrix = create_matrix_from_judgments(crit_judgments, criteria, TFN)
    model.set_comparison_matrix("Goal", crit_matrix)

    # Cost vs its Sub-criteria
    sub_crit_judgments = {("Purchase Price", "Maintenance"): 3}
    sub_crit_matrix = create_matrix_from_judgments(sub_crit_judgments, sub_criteria_cost, TFN)
    model.set_comparison_matrix("Cost", sub_crit_matrix)

    # Set performance scores (leaf nodes are 'Purchase Price', 'Maintenance', 'Quality')
    for alt_name in alternatives:
        alt = model.get_alternative(alt_name)
        alt.set_performance_score("Purchase Price", TFN(0.6, 0.7, 0.8))
        alt.set_performance_score("Maintenance", TFN(0.3, 0.4, 0.5))
        alt.set_performance_score("Quality", TFN(0.8, 0.9, 1.0))

    return model

# --- Tests for Node Class ---

def test_node_creation_and_parenting():
    """Test that nodes are created and linked correctly."""
    parent = Node("P1")
    child = Node("C1")
    parent.add_child(child)

    assert child in parent.children
    assert child.parent == parent
    assert parent.is_leaf is False
    assert child.is_leaf is True

# --- Tests for Hierarchy (AHPModel) Class ---

def test_hierarchy_initialization():
    """Test basic initialization of the Hierarchy class."""
    goal_node = Node("MyGoal")
    model = Hierarchy[Crisp](goal_node, number_type=Crisp)

    assert model.root == goal_node
    assert model.number_type == Crisp
    assert not model.alternatives # Starts empty

def test_add_and_get_alternative(sample_tfn_model):
    """Test adding and retrieving alternatives."""
    assert len(sample_tfn_model.alternatives) == 2

    alt_a = sample_tfn_model.get_alternative("Option A")
    assert isinstance(alt_a, Alternative)
    assert alt_a.name == "Option A"

    # Test getting a non-existent alternative
    with pytest.raises(ValueError, match="Alternative 'Non-existent' not found."):
        sample_tfn_model.get_alternative("Non-existent")

def test_find_node(sample_tfn_model):
    """Test the internal _find_node helper method."""
    goal_node = sample_tfn_model._find_node("Goal")
    cost_node = sample_tfn_model._find_node("Cost")
    purchase_node = sample_tfn_model._find_node("Purchase Price")
    non_existent_node = sample_tfn_model._find_node("FakeNode")

    assert goal_node == sample_tfn_model.root
    assert cost_node is not None and cost_node.id == "Cost"
    assert purchase_node is not None and purchase_node.parent == cost_node
    assert non_existent_node is None

def test_set_comparison_matrix_validation(sample_tfn_model):
    """Test that setting a matrix with incorrect dimensions raises an error."""
    # Trying to set a 3x3 matrix for the "Cost" node, which has 2 children
    invalid_matrix = np.eye(3)
    with pytest.raises(ValueError, match="Matrix dimensions .* do not match the number of children"):
        sample_tfn_model.set_comparison_matrix("Cost", invalid_matrix)

# --- Tests for Calculation Orchestration ---

def test_calculate_weights(sample_tfn_model):
    """Test that the calculate_weights method runs and populates weights."""
    # Before calculation, weights should be None
    cost_node = sample_tfn_model._find_node("Cost")
    assert cost_node.local_weight is None
    assert cost_node.global_weight is None

    # Run calculation
    sample_tfn_model.calculate_weights()

    # After calculation, weights should be populated and be of the correct type
    assert isinstance(cost_node.local_weight, TFN)
    assert isinstance(cost_node.global_weight, TFN)
    assert cost_node.global_weight == cost_node.local_weight # Since its parent is the root

    purchase_node = sample_tfn_model._find_node("Purchase Price")
    assert isinstance(purchase_node.global_weight, TFN)
    # Global weight of a sub-criterion should be its local weight * parent's global weight
    expected_global = purchase_node.local_weight * cost_node.global_weight
    assert purchase_node.global_weight == expected_global

def test_calculate_alternative_scores(sample_tfn_model):
    """Test that the calculate_alternative_scores method runs and populates scores."""
    # First, we must calculate weights
    sample_tfn_model.calculate_weights()

    # Before score calculation, overall_score is None
    alt_a = sample_tfn_model.get_alternative("Option A")
    assert alt_a.overall_score is None

    # Run score calculation
    sample_tfn_model.calculate_alternative_scores()

    # After calculation, scores should be populated
    assert isinstance(alt_a.overall_score, TFN)
    assert alt_a.overall_score.m > 0 # Should be a positive score

    # Check that intermediate node scores were also calculated
    assert "Cost" in alt_a.node_scores
    assert isinstance(alt_a.node_scores["Cost"], TFN)

def test_get_rankings(sample_tfn_model):
    """Test that get_rankings produces a sorted list of tuples."""
    sample_tfn_model.calculate_weights()
    sample_tfn_model.calculate_alternative_scores()

    rankings = sample_tfn_model.get_rankings()

    assert isinstance(rankings, list)
    assert len(rankings) == 2
    assert isinstance(rankings[0], tuple)

    # Check that the list is sorted by score (descending)
    score1, score2 = rankings[0][1], rankings[1][1]
    assert score1 >= score2
