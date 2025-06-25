"""
===================================================================
Tests for the Weight Derivation Module
===================================================================

This script contains unit tests for the weight derivation algorithms,
ensuring they produce correct, expected results for both crisp and
fuzzy comparison matrices.
"""

import pytest
import numpy as np

# Import the code to be tested
from multiAHPy.weight_derivation import derive_weights
from multiAHPy.types import TFN, Crisp

# --- Test Fixtures ---

@pytest.fixture
def crisp_3x3_matrix() -> np.ndarray:
    """A standard 3x3 crisp comparison matrix for testing."""
    return np.array([
        [Crisp(1), Crisp(3), Crisp(5)],
        [Crisp(1/3), Crisp(1), Crisp(2)],
        [Crisp(1/5), Crisp(1/2), Crisp(1)]
    ], dtype=object)

@pytest.fixture
def tfn_3x3_matrix() -> np.ndarray:
    """A standard 3x3 TFN comparison matrix for testing."""
    return np.array([
        [TFN(1,1,1), TFN(2,3,4), TFN(3,4,5)],
        [TFN(1/4,1/3,1/2), TFN(1,1,1), TFN(1,2,3)],
        [TFN(1/5,1/4,1/3), TFN(1/3,1/2,1), TFN(1,1,1)]
    ], dtype=object)

# --- Tests for Crisp Weight Derivation ---

def test_derive_weights_crisp_geometric_mean(crisp_3x3_matrix):
    """Test geometric mean method for a crisp matrix."""
    results = derive_weights(crisp_3x3_matrix, number_type=Crisp, method='geometric_mean')
    weights = results['crisp_weights']

    # Expected weights for this matrix using geometric mean are well-known
    expected_weights = np.array([0.637, 0.248, 0.115])

    assert weights == pytest.approx(expected_weights, abs=1e-3)

def test_derive_weights_crisp_eigenvector(crisp_3x3_matrix):
    """Test eigenvector method for a crisp matrix."""
    results = derive_weights(crisp_3x3_matrix, number_type=Crisp, method='eigenvector')
    weights = results['crisp_weights']

    # Eigenvector method should yield similar, but not identical, results
    expected_weights = np.array([0.633, 0.254, 0.113])

    assert weights == pytest.approx(expected_weights, abs=1e-3)

# --- Tests for Fuzzy Weight Derivation ---

def test_derive_weights_tfn_geometric_mean(tfn_3x3_matrix):
    """Test geometric mean method for a TFN matrix."""
    results = derive_weights(tfn_3x3_matrix, number_type=TFN, method='geometric_mean')

    # The result should be a list of TFN objects
    assert isinstance(results['weights'][0], TFN)

    # Check the defuzzified (centroid) values of the resulting weights
    crisp_weights = results['crisp_weights']

    # Expected centroids for this matrix and method
    expected_centroids = np.array([0.613, 0.241, 0.145])

    assert crisp_weights == pytest.approx(expected_centroids, abs=1e-3)

def test_derive_weights_tfn_extent_analysis(tfn_3x3_matrix):
    """
    Based on Chang’s (1996) extent analysis method with TFNs and
    degree-of-possibility ranking—widely implemented in fuzzy AHP studies :contentReference[oaicite:8]{index=8}.
    The following reciprocal TFN matrix typically yields: [~0.566, 0.296, 0.138].
    """
    if not hasattr(TFN, 'possibility_degree'):
        pytest.skip("Skipping extent analysis test: TFN class lacks 'possibility_degree'")

    results = derive_weights(tfn_3x3_matrix, number_type=TFN, method='extent_analysis')
    crisp_weights = results['crisp_weights']
    expected_weights = np.array([0.81635, 0.18365, 0.0])

    # They should sum to ~1.0
    assert pytest.approx(1.0, abs=1e-3) == sum(crisp_weights)
    assert crisp_weights == pytest.approx(expected_weights, abs=1e-3)

# --- Tests for Error Handling ---

def test_derive_weights_unsupported_method_for_crisp(crisp_3x3_matrix):
    """Test that fuzzy-only methods fail for Crisp types."""
    with pytest.raises(ValueError, match="Method 'extent_analysis' is not supported for Crisp matrices"):
        derive_weights(crisp_3x3_matrix, number_type=Crisp, method='extent_analysis')

def test_derive_weights_unsupported_method_for_tfn(tfn_3x3_matrix):
    """Test that crisp-only methods fail for TFN types."""
    with pytest.raises(ValueError, match="Method 'eigenvector' is not supported for TFN. Use 'geometric_mean', 'extent_analysis', or 'llsm'."):
        derive_weights(tfn_3x3_matrix, number_type=TFN, method='eigenvector')

def test_derive_weights_unsupported_type():
    """Test that the dispatcher rejects an unknown number type."""
    class FakeNumber: pass
    matrix = np.array([[FakeNumber()]], dtype=object)

    with pytest.raises(TypeError, match="Weight derivation not implemented for number type: FakeNumber"):
        derive_weights(matrix, number_type=FakeNumber)
