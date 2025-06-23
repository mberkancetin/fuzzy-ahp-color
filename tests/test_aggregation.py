"""
===================================================================
Tests for the Aggregation Module
===================================================================

This script contains unit tests for the matrix aggregation functionalities
using the pytest framework.

To run tests, navigate to the root directory (fuzzy-ahp-color/) and run:
$ pytest
"""

import pytest
import numpy as np

# Import the code to be tested
from multiAHPy.aggregation import aggregate_matrices
from multiAHPy.types import TFN, Crisp

# --- Test Fixtures: Reusable test data ---

@pytest.fixture
def sample_tfn_matrices():
    """Provides a set of three valid TFN matrices for testing."""
    m1 = np.array([
        [TFN(1,1,1), TFN(2,3,4)],
        [TFN(1/4, 1/3, 1/2), TFN(1,1,1)]
    ], dtype=object)

    m2 = np.array([
        [TFN(1,1,1), TFN(4,5,6)],
        [TFN(1/6, 1/5, 1/4), TFN(1,1,1)]
    ], dtype=object)

    m3 = np.array([
        [TFN(1,1,1), TFN(1,1,1)],
        [TFN(1,1,1), TFN(1,1,1)]
    ], dtype=object)

    return [m1, m2, m3]

@pytest.fixture
def sample_crisp_matrices():
    """Provides a set of two valid Crisp matrices for testing."""
    m1 = np.array([
        [Crisp(1), Crisp(4)],
        [Crisp(0.25), Crisp(1)]
    ], dtype=object)

    m2 = np.array([
        [Crisp(1), Crisp(2)],
        [Crisp(0.5), Crisp(1)]
    ], dtype=object)

    return [m1, m2]

# --- Tests for aggregate_matrices function ---

def test_arithmetic_mean_tfn(sample_tfn_matrices):
    """Test arithmetic mean aggregation for TFNs."""
    aggregated = aggregate_matrices(sample_tfn_matrices, method="arithmetic")

    # Expected result for cell (0, 1):
    # l = (2+4+1)/3 = 7/3, m = (3+5+1)/3 = 9/3, u = (4+6+1)/3 = 11/3
    expected_l = 7/3
    expected_m = 3.0
    expected_u = 11/3

    result_cell = aggregated[0, 1]

    assert isinstance(result_cell, TFN)
    assert result_cell.l == pytest.approx(expected_l)
    assert result_cell.m == pytest.approx(expected_m)
    assert result_cell.u == pytest.approx(expected_u)

def test_geometric_mean_crisp(sample_crisp_matrices):
    """Test geometric mean aggregation for Crisp numbers."""
    aggregated = aggregate_matrices(sample_crisp_matrices, method="geometric")

    # Expected result for cell (0, 1): sqrt(4 * 2) = sqrt(8)
    expected_val = np.sqrt(8)
    result_cell = aggregated[0, 1]

    assert isinstance(result_cell, Crisp)
    assert result_cell.value == pytest.approx(expected_val)

    # Check reciprocity
    reciprocal_cell = aggregated[1, 0]
    assert reciprocal_cell.value == pytest.approx(1 / expected_val)

def test_median_aggregation(sample_tfn_matrices):
    """Test median aggregation method."""
    aggregated = aggregate_matrices(sample_tfn_matrices, method="median")

    # For cell (0, 1), the values are (2,3,4), (4,5,6), (1,1,1)
    # Median of l's (2,4,1) is 2
    # Median of m's (3,5,1) is 3
    # Median of u's (4,6,1) is 4
    expected_tfn = TFN(2, 3, 4)
    result_cell = aggregated[0, 1]

    assert result_cell == expected_tfn

def test_min_max_aggregation(sample_tfn_matrices):
    """Test min-max aggregation method."""
    aggregated = aggregate_matrices(sample_tfn_matrices, method="min_max")

    # For cell (0,1), l's are (2,4,1), m's are (3,5,1), u's are (4,6,1)
    # min(l) = 1, mean(m) = 3, max(u) = 6
    expected_tfn = TFN(1, 3, 6)
    result_cell = aggregated[0, 1]

    assert result_cell == expected_tfn

def test_weighted_aggregation(sample_crisp_matrices):
    """Test aggregation with expert weights."""
    weights = [0.2, 0.8] # Give more weight to the second expert
    aggregated = aggregate_matrices(sample_crisp_matrices, method="arithmetic", expert_weights=weights)

    # Expected result for cell (0,1): (4 * 0.2) + (2 * 0.8) = 0.8 + 1.6 = 2.4
    expected_val = 2.4
    result_cell = aggregated[0, 1]

    assert result_cell.value == pytest.approx(expected_val)

def test_aggregation_with_invalid_method():
    """Test that an invalid method raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        aggregate_matrices([np.eye(2)], method="invalid_method")

def test_aggregation_with_mismatched_shapes(sample_crisp_matrices):
    """Test that matrices with different shapes raise a ValueError."""
    mismatched_list = [sample_crisp_matrices[0], np.eye(3)]
    with pytest.raises(ValueError, match="All matrices must have the same dimensions"):
        aggregate_matrices(mismatched_list)
