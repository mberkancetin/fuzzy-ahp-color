"""
===================================================================
Tests for the Consistency Module
===================================================================

This script contains unit tests for the matrix consistency analysis
functionalities using the pytest framework.

To run tests, navigate to the root directory and run:
$ pytest
"""

import pytest
import numpy as np

from multiAHPy.consistency import Consistency
from multiAHPy.types import TFN, Crisp
from multiAHPy.model import Hierarchy, Node

# ==============================================================================
# Test Fixtures: Reusable and Correct Test Data
# ==============================================================================

@pytest.fixture
def perfectly_consistent_crisp_matrix() -> np.ndarray:
    """A perfectly consistent 3x3 crisp matrix."""
    return np.array([
        [Crisp(1), Crisp(2), Crisp(4)],
        [Crisp(0.5), Crisp(1), Crisp(2)],
        [Crisp(0.25), Crisp(0.5), Crisp(1)]
    ], dtype=object)

@pytest.fixture
def saaty_inconsistent_crisp_matrix() -> np.ndarray:
    """
    A classic, clearly inconsistent 3x3 matrix from Saaty's literature.
    The expected CR is approximately 0.035 for the geometric mean weight method.
    Note: Different weight derivation methods can yield slightly different CRs.
    """
    return np.array([
        [Crisp(1), Crisp(1/4), Crisp(4)],
        [Crisp(4), Crisp(1), Crisp(9)],
        [Crisp(1/4), Crisp(1/9), Crisp(1)]
    ], dtype=object)

@pytest.fixture
def very_inconsistent_tfn_matrix() -> np.ndarray:
    return np.array([
        [TFN(1,1,1), TFN(4,5,6), TFN(1/7, 1/6, 1/5)],
        [TFN(1/6, 1/5, 1/4), TFN(1,1,1), TFN(8,9,9)],
        [TFN(5,6,7), TFN(1/9, 1/9, 1/8), TFN(1,1,1)]
    ], dtype=object)

@pytest.fixture
def inconsistent_model(very_inconsistent_tfn_matrix: np.ndarray) -> Hierarchy:
    """Creates a full Hierarchy object with a very inconsistent matrix."""
    goal_node = Node("Goal", "Test Goal")
    model = Hierarchy[TFN](goal_node, number_type=TFN)

    criteria = ["C1", "C2", "C3"]
    for name in criteria:
        goal_node.add_child(Node(name))

    model.set_comparison_matrix("Goal", very_inconsistent_tfn_matrix)
    return model

# ==============================================================================
# Tests for Consistency Class Static Methods
# ==============================================================================

def test_get_random_index():
    """Tests the retrieval of RI values for various matrix sizes."""
    assert Consistency._get_random_index(3) == 0.52
    assert Consistency._get_random_index(9) == 1.45
    assert Consistency._get_random_index(1) == 0.00
    assert Consistency._get_random_index(20) == 1.60 # Test fallback

def test_cr_for_perfectly_consistent_matrix(perfectly_consistent_crisp_matrix: np.ndarray):
    """A perfectly consistent matrix must have a CR of approximately 0."""
    cr = Consistency.calculate_saaty_cr(perfectly_consistent_crisp_matrix)
    assert cr == pytest.approx(0.0, abs=1e-9)

def test_cr_for_2x2_matrix():
    """Any 2x2 reciprocal matrix is always perfectly consistent (CR=0)."""
    matrix_2x2 = np.array([[Crisp(1), Crisp(5)], [Crisp(0.2), Crisp(1)]], dtype=object)
    cr = Consistency.calculate_saaty_cr(matrix_2x2)
    assert cr == 0.0

def test_cr_for_saaty_inconsistent_example(saaty_inconsistent_crisp_matrix: np.ndarray):
    """
    Test the CR calculation against a known inconsistent example using the
    geometric mean weight approximation for lambda_max.
    For this specific matrix and method, CR should be ~0.035.
    """
    cr = Consistency.calculate_saaty_cr(saaty_inconsistent_crisp_matrix)
    assert cr == pytest.approx(0.03547, abs=1e-4)

def test_cr_for_very_inconsistent_tfn_matrix(very_inconsistent_tfn_matrix: np.ndarray):
    """Test that a clearly inconsistent TFN matrix yields a high CR."""
    cr = Consistency.calculate_saaty_cr(very_inconsistent_tfn_matrix, consistency_method='centroid')
    # For a 3x3 matrix, CR > 0.05 is inconsistent, and this one should be much higher.
    assert cr > 0.1

def test_get_consistency_recommendations(inconsistent_model: Hierarchy):
    """Tests the recommendation generation for an inconsistent matrix."""
    recommendations = Consistency.get_consistency_recommendations(inconsistent_model, "Goal")

    assert isinstance(recommendations, list)
    # Expect 3 lines: header, which pair is inconsistent, and suggested value.
    assert len(recommendations) >= 3

    # Check that the recommendation strings contain the expected content.
    assert "inconsistent (CR =" in recommendations[0]
    assert "judgment appears to be between" in recommendations[1]
    assert "should be closer to" in recommendations[2]

def test_check_model_consistency(inconsistent_model: Hierarchy):
    """
    Tests the main function that checks all matrices in a model.
    It should correctly identify the inconsistent matrix.
    """
    results = Consistency.check_model_consistency(inconsistent_model, saaty_cr_threshold=0.1)

    assert "Goal" in results
    goal_results = results["Goal"]

    # More Pythonic assertions
    assert not goal_results["is_consistent"]  # Instead of == False
    assert goal_results["saaty_cr"] > 0.1
    assert goal_results["matrix_size"] == 3

def test_cr_for_saaty_inconsistent_example():
    """Test CR calculation against a known inconsistent example."""
    matrix = np.array([
        [Crisp(1), Crisp(1/4), Crisp(4)],
        [Crisp(4), Crisp(1), Crisp(9)],
        [Crisp(1/4), Crisp(1/9), Crisp(1)]
    ], dtype=object)

    # The geometric mean method produces this CR for this matrix.
    # The bug was in my expected value, not your code.
    expected_cr = 0.03547

    cr = Consistency.calculate_saaty_cr(matrix)
    assert cr == pytest.approx(expected_cr, abs=1e-4)

def test_check_model_with_inconsistent_matrix():
    """
    Tests that the model check correctly identifies a matrix as inconsistent
    when its CR exceeds a STRICT threshold.
    """
    inconsistent_matrix = np.array([
        [Crisp(1), Crisp(2), Crisp(8)],     # C1 vs C2: 2, C1 vs C3: 8
        [Crisp(1/2), Crisp(1), Crisp(2)],   # C2 vs C1: 1/2 ✓, C2 vs C3: 2
        [Crisp(1/8), Crisp(3), Crisp(1)]    # C3 vs C1: 1/8 ✓, C3 vs C2: 3 (should be 1/2!)
    ], dtype=object)

    goal_node = Node("Goal", "Test Goal")
    model = Hierarchy[Crisp](goal_node, number_type=Crisp)
    for name in ["C1", "C2", "C3"]:
        goal_node.add_child(Node(name))
    model.set_comparison_matrix("Goal", inconsistent_matrix)

    results = Consistency.check_model_consistency(model, saaty_cr_threshold=0.1)

    assert results["Goal"]["is_consistent"] == False

def test_check_model_with_consistent_matrix():
    """
    Tests that the model check correctly identifies a matrix as consistent
    when its CR is below a standard threshold.
    """
    consistent_matrix = np.array([
        [Crisp(1), Crisp(1/4), Crisp(4)],
        [Crisp(4), Crisp(1), Crisp(9)],
        [Crisp(1/4), Crisp(1/9), Crisp(1)]
    ], dtype=object)

    goal_node = Node("Goal", "Test Goal")
    model = Hierarchy[Crisp](goal_node, number_type=Crisp)
    for name in ["C1", "C2", "C3"]:
        goal_node.add_child(Node(name))
    model.set_comparison_matrix("Goal", consistent_matrix)

    results = Consistency.check_model_consistency(model, saaty_cr_threshold=0.1)

    # More robust assertions
    assert "Goal" in results
    assert "is_consistent" in results["Goal"]
    assert results["Goal"]["is_consistent"] == True
    assert results["Goal"]["saaty_cr"] < 0.1

def test_cr_saaty_actual_example():
    """
    Based on actual Saaty examples from literature.
    """
    saaty_matrix = np.array([
        [Crisp(1), Crisp(1/4), Crisp(1/2)],
        [Crisp(4), Crisp(1), Crisp(3)],
        [Crisp(2), Crisp(1/3), Crisp(1)]
    ], dtype=object)

    cr = Consistency.calculate_saaty_cr(saaty_matrix)
    print(f"CR: {cr}")
    expected_cr = 0.01759106470156774
    assert abs(cr - expected_cr) < 1e-3

def test_cr_for_saaty_example1():
    """
    Test the CR calculation against a known inconsistent example.
    This matrix has significant inconsistency with CR > 0.1
    """
    saaty_inconsistent_crisp_matrix = np.array([
        [Crisp(1), Crisp(3), Crisp(7)],
        [Crisp(1/3), Crisp(1), Crisp(9)],  # This 9 creates inconsistency
        [Crisp(1/7), Crisp(1/9), Crisp(1)]
    ], dtype=object)

    cr = Consistency.calculate_saaty_cr(saaty_inconsistent_crisp_matrix)
    # This should give CR > 0.1
    assert cr > 0.1

def test_cr_for_consistent_example2():
    """
    Test the CR calculation against a nearly consistent matrix.
    This matrix is actually quite consistent.
    """
    consistent_crisp_matrix = np.array([
        [Crisp(1), Crisp(1/4), Crisp(4)],
        [Crisp(4), Crisp(1), Crisp(9)],
        [Crisp(1/4), Crisp(1/9), Crisp(1)]
    ], dtype=object)

    cr = Consistency.calculate_saaty_cr(consistent_crisp_matrix)
    # This matrix is actually consistent with CR ≈ 0.035
    assert cr == pytest.approx(0.035, abs=0.01)
    assert cr < 0.1  # Should be consistent

def test_manual_verification():
    """
    Manually verify the consistency calculation with a known example.
    """
    # Use a simple matrix where we can calculate CR by hand
    matrix = np.array([
        [Crisp(1), Crisp(3), Crisp(7)],
        [Crisp(1/3), Crisp(1), Crisp(3)],
        [Crisp(1/7), Crisp(1/3), Crisp(1)]
    ], dtype=object)

    # Manual calculation
    crisp_matrix = np.array([
        [1, 3, 7],
        [1/3, 1, 3],
        [1/7, 1/3, 1]
    ])

    print(f"Matrix:\n{crisp_matrix}")

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(crisp_matrix)
    lambda_max = np.max(np.real(eigenvalues))
    print(f"Lambda max: {lambda_max}")

    # Calculate CI
    n = 3
    ci = (lambda_max - n) / (n - 1)
    print(f"CI: {ci}")

    # Calculate CR (RI for n=3 is 0.52)
    ri = 0.52
    cr_manual = ci / ri
    print(f"Manual CR: {cr_manual}")

    # Compare with your function
    cr_function = Consistency.calculate_saaty_cr(matrix)
    print(f"Function CR: {cr_function}")

    assert abs(cr_manual - cr_function) < 0.001
