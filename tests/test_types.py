"""
===================================================================
Tests for the Types Module
===================================================================

This script contains unit tests for the NumericType classes (Crisp, TFN,
TrFN, GFN) to ensure their arithmetic and special methods are mathematically
correct and robust.
"""

import pytest
import numpy as np

from multiAHPy.types import Crisp, TFN, TrFN, GFN

# --- Test Data Fixtures ---

@pytest.fixture
def tfn1() -> TFN:
    return TFN(1, 2, 3)

@pytest.fixture
def tfn2() -> TFN:
    return TFN(2, 4, 6)

@pytest.fixture
def crisp1() -> Crisp:
    return Crisp(5.0)

# ==============================================================================
# Tests for Crisp Class
# ==============================================================================

def test_crisp_arithmetic(crisp1):
    assert (crisp1 + 2).value == 7.0
    assert (3 + crisp1).value == 8.0 # Test radd
    assert (crisp1 - 1).value == 4.0
    assert (10 - crisp1).value == 5.0 # Test rsub
    assert (crisp1 * 3).value == 15.0
    assert (crisp1 / 2).value == 2.5
    assert (10 / crisp1).value == 2.0 # Test rtruediv
    assert (crisp1 ** 2).value == 25.0

def test_crisp_identities():
    assert Crisp.neutral_element().value == 0.0
    assert Crisp.multiplicative_identity().value == 1.0

def test_crisp_inverse(crisp1):
    assert crisp1.inverse().value == 0.2

def test_crisp_comparisons():
    assert Crisp(5) > Crisp(4)
    assert Crisp(5) >= 5
    assert Crisp(5) == 5.0
    assert not (Crisp(5) < 4)

# ==============================================================================
# Tests for TFN Class
# ==============================================================================

def test_tfn_initialization():
    """Test that invalid TFN values raise an error."""
    with pytest.raises(ValueError, match="TFN values must satisfy l <= m <= u"):
        TFN(5, 2, 3)

def test_tfn_addition(tfn1, tfn2):
    result = tfn1 + tfn2
    assert result == TFN(3, 6, 9)
    assert isinstance(result, TFN)
    assert result.l <= result.m <= result.u

    # Test with scalar
    result_scalar = tfn1 + 10
    assert result_scalar == TFN(11, 12, 13)

    # Test reflected add
    result_radd = 10 + tfn1
    assert result_radd == TFN(11, 12, 13)

def test_tfn_multiplication(tfn1, tfn2):
    result = tfn1 * tfn2
    assert result == TFN(2, 8, 18)
    assert isinstance(result, TFN)
    assert result.l <= result.m <= result.u

    result_scalar = tfn1 * 2
    assert result_scalar == TFN(2, 4, 6)

    result_rmul = 2 * tfn1
    assert result_rmul == TFN(2, 4, 6)

def test_tfn_division(tfn2):
    # tfn2 = (2, 4, 6)
    tfn_half = TFN(1, 2, 3)
    result = tfn2 / tfn_half

    # Expected l = 2 / 3
    # Expected m = 4 / 2 = 2
    # Expected u = 6 / 1 = 6

    assert result.l == pytest.approx(2/3)
    assert result.m == pytest.approx(2.0)
    assert result.u == pytest.approx(6.0)
    assert result.l <= result.m <= result.u
    assert isinstance(result, TFN)

    # TFN / scalar
    result_scalar = tfn2 / 2
    assert result_scalar == TFN(1, 2, 3)

    # scalar / TFN
    result_rdiv = 12 / tfn2
    assert result_rdiv == TFN(2, 3, 6)

def test_tfn_power(tfn1):
    result = tfn1 ** 2
    assert result == TFN(1, 4, 9)

    result_neg = tfn1 ** -1
    assert result_neg == tfn1.inverse()

def test_tfn_inverse(tfn1):
    inv = tfn1.inverse()
    assert inv.l == pytest.approx(1/3)
    assert inv.m == pytest.approx(1/2)
    assert inv.u == pytest.approx(1)
    assert isinstance(inv, TFN)
    assert inv.l <= inv.m <= inv.u

def test_tfn_centroid_and_defuzzify(tfn1):
    expected_centroid = (1 + 2 + 3) / 3.0
    assert tfn1.defuzzify() == pytest.approx(expected_centroid)

def test_tfn_from_crisp():
    """Test the factory method for creating a TFN from a crisp value."""
    crisp_tfn = TFN.from_crisp(5.0)
    assert crisp_tfn == TFN(5, 5, 5)

# ==============================================================================
# Tests for TrFN and GFN (Abbreviated examples)
# You can expand these with more detailed arithmetic tests if needed.
# ==============================================================================

def test_trfn_initialization():
    """Test Trapezoidal Fuzzy Number creation."""
    trfn = TrFN(1, 3, 5, 7)
    assert trfn.a == 1 and trfn.b == 3 and trfn.c == 5 and trfn.d == 7

    with pytest.raises(ValueError):
        TrFN(1, 5, 3, 7) # Invalid order

def test_trfn_centroid():
    trfn = TrFN(1, 2, 4, 5)
    # Expected: (5^2+4^2+5*4) - (1^2+2^2+1*2) / 3*((5+4)-(1+2))
    # num = (25+16+20) - (1+4+2) = 61 - 7 = 54
    # den = 3 * (9 - 3) = 18
    # result = 54 / 18 = 3.0
    assert trfn.defuzzify() == pytest.approx(3.0)

def test_gfn_arithmetic():
    """Test Gaussian Fuzzy Number addition."""
    gfn1 = GFN(m=10, sigma=3)
    gfn2 = GFN(m=20, sigma=4)

    # Test addition: m = 10+20=30, sigma = sqrt(3^2+4^2)=sqrt(25)=5
    result = gfn1 + gfn2
    assert isinstance(result, GFN)
    assert result.m == 30
    assert result.sigma == 5
