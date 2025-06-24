"""
===================================================================
Tests for the Validation Module
===================================================================

This script contains unit tests for the validation functionalities, ensuring
that invalid model structures and matrices are correctly identified.
"""

import pytest
import numpy as np

from multiAHPy.validation import Validation
from multiAHPy.types import TFN, Crisp
from multiAHPy.model import Hierarchy, Node

# --- Test Fixtures: Reusable valid and invalid components ---

@pytest.fixture
def valid_crisp_matrix() -> np.ndarray:
    """A valid, 3x3 reciprocal crisp matrix."""
    return np.array([
        [Crisp(1), Crisp(3), Crisp(5)],
        [Crisp(1/3), Crisp(1), Crisp(2)],
        [Crisp(1/5), Crisp(1/2), Crisp(1)]
    ], dtype=object)

@pytest.fixture
def non_square_matrix() -> np.ndarray:
    """A non-square matrix, which is invalid."""
    return np.array([
        [Crisp(1), Crisp(2)],
        [Crisp(0.5), Crisp(1)],
        [Crisp(1), Crisp(1)]
    ], dtype=object)

@pytest.fixture
def bad_diagonal_matrix() -> np.ndarray:
    """A matrix with an incorrect diagonal element."""
    return np.array([
        [Crisp(1), Crisp(2)],
        [Crisp(0.5), Crisp(2)] # Diagonal should be 1
    ], dtype=object)

@pytest.fixture
def non_reciprocal_matrix() -> np.ndarray:
    """A matrix that fails the reciprocity check."""
    return np.array([
        [Crisp(1), Crisp(2)],
        [Crisp(0.8), Crisp(1)] # Should be 0.5
    ], dtype=object)

@pytest.fixture
def sample_model() -> Hierarchy:
    """Provides a basic, correctly structured model for testing."""
    goal_node = Node("Goal", "Test Goal")
    model = Hierarchy[Crisp](goal_node, number_type=Crisp)

    c1 = Node("C1")
    c2 = Node("C2")
    goal_node.add_child(c1)
    goal_node.add_child(c2)

    model.add_alternative("Alt A")
    return model

# --- Tests for Individual Matrix Validation Functions ---

def test_validate_matrix_properties_on_valid_matrix(valid_crisp_matrix):
    """A valid matrix should produce no error strings."""
    errors = Validation.validate_matrix_properties(valid_crisp_matrix)
    assert not errors # The list of errors should be empty

def test_validate_matrix_dimensions(non_square_matrix, valid_crisp_matrix):
    """Test dimension validation."""
    errors = Validation.validate_matrix_dimensions(non_square_matrix)
    assert len(errors) == 1
    assert "2D square" in errors[0]

    # Test with expected size mismatch
    errors_size = Validation.validate_matrix_dimensions(valid_crisp_matrix, expected_size=4)
    assert len(errors_size) == 1
    assert "expected size 4" in errors_size[0]

def test_validate_matrix_diagonal(bad_diagonal_matrix):
    """Test diagonal validation."""
    errors = Validation.validate_matrix_diagonal(bad_diagonal_matrix)
    assert len(errors) == 1
    assert "Diagonal element at (1,1) is not 1" in errors[0]

def test_validate_matrix_reciprocity(non_reciprocal_matrix):
    """Test reciprocity validation."""
    errors = Validation.validate_matrix_reciprocity(non_reciprocal_matrix)
    assert len(errors) == 1
    assert "Reciprocity failed between (0,1) and (1,0)" in errors[0]

# --- Tests for Model-Level Validation ---

def test_validate_hierarchy_completeness_success(sample_model, valid_crisp_matrix):
    """Test a complete hierarchy should pass validation."""
    # Complete the model by adding the required matrix
    # Note: size of valid_crisp_matrix is 3, but model has 2 criteria. Create a 2x2.
    matrix_2x2 = np.array([[Crisp(1), Crisp(4)], [Crisp(0.25), Crisp(1)]], dtype=object)
    sample_model.set_comparison_matrix("Goal", matrix_2x2)

    errors = Validation.validate_hierarchy_completeness(sample_model)
    assert not errors # Should be an empty list

def test_validate_hierarchy_completeness_missing_matrix(sample_model):
    """Test that a missing comparison matrix is detected."""
    # The sample_model fixture is created without a matrix for the 'Goal' node
    errors = Validation.validate_hierarchy_completeness(sample_model)
    assert len(errors) == 1
    assert "Node 'Goal' is a parent but has no comparison matrix set" in errors[0]

def test_validate_performance_scores_success(sample_model):
    """Test that a model with all performance scores set passes validation."""
    # Add scores for the two leaf nodes (C1, C2) for the one alternative
    alt = sample_model.get_alternative("Alt A")
    alt.set_performance_score("C1", Crisp(0.8))
    alt.set_performance_score("C2", Crisp(0.2))

    errors = Validation.validate_performance_scores(sample_model)
    assert not errors

def test_validate_performance_scores_missing_score(sample_model):
    """Test that a missing performance score is detected."""
    # Only set the score for C1, leaving C2 missing
    alt = sample_model.get_alternative("Alt A")
    alt.set_performance_score("C1", Crisp(0.8))

    errors = Validation.validate_performance_scores(sample_model)
    assert len(errors) == 1
    assert "Alternative 'Alt A' is missing a performance score for leaf node 'C2'" in errors[0]

# --- Test for the Main Wrapper Function ---

def test_run_all_validations_with_multiple_errors():
    """Test the main wrapper function aggregates errors from all checks."""
    # Create a model with multiple problems
    goal_node = Node("Goal")
    model = Hierarchy[Crisp](goal_node, number_type=Crisp)
    goal_node.add_child(Node("C1"))
    goal_node.add_child(Node("C2"))
    model.add_alternative("Alt A")
    # Problem 1: Missing matrix for 'Goal'
    # Problem 2: Missing performance score for 'Alt A' vs 'C1' and 'C2'

    all_errors = Validation.run_all_validations(model)

    assert len(all_errors["hierarchy_completeness"]) == 1
    assert "Node 'Goal' is a parent but has no comparison matrix" in all_errors["hierarchy_completeness"][0]

    # matrix_properties should be empty because there's no matrix to check
    assert not all_errors["matrix_properties"]

    assert len(all_errors["performance_scores"]) == 2
    assert "missing a performance score for leaf node 'C1'" in all_errors["performance_scores"][0]
    assert "missing a performance score for leaf node 'C2'" in all_errors["performance_scores"][1]
