"""
===================================================================
Tests for the Defuzzification Module
===================================================================

This script contains unit tests for the defuzzification functionalities,
ensuring that different numeric types are correctly converted to crisp values
using various standard methods.
"""

import pytest
import numpy as np

# Import the code to be tested
from multiAHPy import defuzzification
from multiAHPy.types import Crisp, TFN, TrFN, GFN, IT2TrFN, IFN

# --- Test Data Fixtures ---

@pytest.fixture
def sample_tfn() -> TFN:
    """A standard Triangular Fuzzy Number for testing."""
    return TFN(l=2, m=4, u=8)

@pytest.fixture
def sample_trfn() -> TrFN:
    """A standard Trapezoidal Fuzzy Number for testing."""
    return TrFN(a=1, b=3, c=5, d=9)

@pytest.fixture
def sample_gfn() -> GFN:
    """A standard Gaussian Fuzzy Number for testing."""
    return GFN(m=5, sigma=1)

@pytest.fixture
def sample_crisp() -> Crisp:
    """A standard Crisp number for testing."""
    return Crisp(7.5)

# --- Tests for TFN Defuzzification ---

def test_tfn_centroid(sample_tfn):
    """Test the centroid method for TFN."""
    # Expected: (2 + 4 + 8) / 3 = 14 / 3
    expected = 14 / 3
    result = TFN.defuzzify(sample_tfn, method='centroid')
    assert result == pytest.approx(expected)

def test_tfn_pessimistic(sample_tfn):
    """Test the pessimistic (lower bound) method for TFN."""
    expected = 2.0
    # Assuming your defuzzification class supports this method
    result = TFN.defuzzify(sample_tfn, method='pessimistic')
    assert result == pytest.approx(expected)

def test_tfn_optimistic(sample_tfn):
    """Test the optimistic (upper bound) method for TFN."""
    expected = 8.0
    result = TFN.defuzzify(sample_tfn, method='optimistic')
    assert result == pytest.approx(expected)

# --- Tests for TrFN Defuzzification ---

def test_trfn_centroid(sample_trfn):
    """Test the centroid method for TrFN."""
    # Expected from formula:
    # num = (d^2+c^2+d*c) - (a^2+b^2+a*b) = (81+25+45) - (1+9+3) = 151 - 13 = 138
    # den = 3*((d+c) - (a+b)) = 3*((9+5) - (1+3)) = 3*(14 - 4) = 30
    # result = 138 / 30 = 4.6
    expected = 4.6
    result = TrFN.defuzzify(sample_trfn, method='centroid')
    assert result == pytest.approx(expected)

def test_trfn_average(sample_trfn):
    """Test the simple average method for TrFN."""
    # Expected: (1 + 3 + 5 + 9) / 4 = 18 / 4 = 4.5
    expected = 4.5
    result = TrFN.defuzzify(sample_trfn, method='average')
    assert result == pytest.approx(expected)

# --- Tests for GFN Defuzzification ---

def test_gfn_centroid(sample_gfn):
    """Test the centroid method for GFN, which is its mean."""
    expected = 5.0
    result = GFN.defuzzify(sample_gfn, method='centroid')
    assert result == pytest.approx(expected)

# --- Tests for Crisp Number Defuzzification ---

def test_crisp_defuzzification(sample_crisp):
    """Test that defuzzifying a Crisp number returns its value."""
    expected = 7.5
    # The method name shouldn't matter for a crisp number
    result_centroid = Crisp.defuzzify(sample_crisp, method='centroid')
    result_other = Crisp.defuzzify(sample_crisp, method='optimistic')

    assert result_centroid == pytest.approx(expected)
    assert result_other == pytest.approx(expected)

# --- Tests for Error Handling ---

def test_unsupported_method(sample_tfn):
    """Test that an unsupported method name raises a ValueError."""
    with pytest.raises(ValueError, match="Method 'invalid_method' not implemented for TFN"):
        TFN.defuzzify(sample_tfn, method='invalid_method')


def test_ifn_entropy_defuzzification():
    """Test the entropy measure for IFNs."""
    # A crisp-like IFN (no fuzziness)
    crisp_ifn = IFN(1.0, 0.0)
    assert IFN.defuzzify(crisp_ifn, method='entropy') == pytest.approx(0.0)

    # A maximally fuzzy IFN (high hesitation)
    max_hesitation_ifn = IFN(0.0, 0.0)
    assert IFN.defuzzify(max_hesitation_ifn, method='entropy') == pytest.approx(1.0)

    # A standard IFN from the fixture
    standard_ifn = IFN(0.6, 0.3) # score = 0.3
    # Expected entropy = 1 - |0.6 - 0.3| = 0.7
    assert IFN.defuzzify(standard_ifn, method='entropy') == pytest.approx(0.7)
