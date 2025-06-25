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

from multiAHPy.types import Crisp, TFN, TrFN, GFN, IFN

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


# ==============================================================================
# Tests for IFN (Intuitionistic Fuzzy Number) Class
# ==============================================================================

@pytest.fixture
def ifn1() -> IFN:
    """Represents a standard IFN, e.g., 'Good'."""
    return IFN(mu=0.6, nu=0.3)

@pytest.fixture
def ifn2() -> IFN:
    """Represents another standard IFN, e.g., 'Fair'."""
    return IFN(mu=0.5, nu=0.4)

def test_ifn_initialization_and_pi():
    """Test valid IFN creation and pi calculation."""
    ifn = IFN(mu=0.7, nu=0.2)
    assert ifn.mu == 0.7
    assert ifn.nu == 0.2
    assert ifn.pi == pytest.approx(0.1)

def test_ifn_invalid_initialization():
    """Test that invalid IFN values raise errors."""
    # Test mu + nu > 1
    with pytest.raises(ValueError, match="Sum of membership and non-membership must not exceed 1"):
        IFN(0.7, 0.4)

    # Test values outside [0, 1]
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        IFN(-0.1, 0.5)
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        IFN(0.5, 1.1)

def test_ifn_addition(ifn1, ifn2):
    """Test standard IFN addition (probabilistic sum)."""
    # ifn1 = (0.6, 0.3); ifn2 = (0.5, 0.4)
    # Expected mu = 0.6 + 0.5 - (0.6 * 0.5) = 1.1 - 0.3 = 0.8
    # Expected nu = 0.3 * 0.4 = 0.12
    result = ifn1 + ifn2
    assert isinstance(result, IFN)
    assert result.mu == pytest.approx(0.8)
    assert result.nu == pytest.approx(0.12)

def test_ifn_multiplication(ifn1, ifn2):
    """Test standard IFN multiplication (probabilistic product)."""
    # ifn1 = (0.6, 0.3); ifn2 = (0.5, 0.4)
    # Expected mu = 0.6 * 0.5 = 0.3
    # Expected nu = 0.3 + 0.4 - (0.3 * 0.4) = 0.7 - 0.12 = 0.58
    result = ifn1 * ifn2
    assert isinstance(result, IFN)
    assert result.mu == pytest.approx(0.3)
    assert result.nu == pytest.approx(0.58)

def test_ifn_power(ifn1):
    """Test raising an IFN to a power."""
    # ifn1 = (0.6, 0.3) raised to power 2
    # Expected mu = 0.6^2 = 0.36
    # Expected nu = 1 - (1 - 0.3)^2 = 1 - 0.7^2 = 1 - 0.49 = 0.51
    result = ifn1 ** 2
    assert result.mu == pytest.approx(0.36)
    assert result.nu == pytest.approx(0.51)

def test_ifn_scale(ifn1):
    """Test scalar multiplication (scaling) of an IFN."""
    # ifn1 = (0.6, 0.3) scaled by 0.5
    # Expected mu = 1 - (1 - 0.6)^0.5 = 1 - 0.4^0.5 = 1 - 0.632 = 0.368
    # Expected nu = 0.3^0.5 = 0.547
    result = ifn1.scale(0.5)
    assert result.mu == pytest.approx(1 - (1 - 0.6)**0.5)
    assert result.nu == pytest.approx(0.3**0.5)

def test_ifn_comparison(ifn1, ifn2):
    """Test the score and accuracy based comparison logic."""
    # Test 1: Simple score comparison
    # ifn1 (0.6, 0.3): score = 0.3
    # ifn2 (0.5, 0.4): score = 0.1
    assert ifn1.score() > ifn2.score()
    assert ifn1 > ifn2
    assert ifn2 < ifn1

    # --- FIX IS HERE: Use valid IFNs for the tie-breaker test ---
    # Test 2: Equal scores, different accuracy
    ifn_low_accuracy = IFN(0.5, 0.2)  # score = 0.3, accuracy = 0.7
    ifn_high_accuracy = IFN(0.6, 0.3) # ifn1, score = 0.3, accuracy = 0.9

    # Verify scores are equal and accuracies are different
    assert ifn_low_accuracy.score() == pytest.approx(ifn_high_accuracy.score())
    assert ifn_high_accuracy.accuracy() > ifn_low_accuracy.accuracy()

    # The one with higher accuracy should be considered "greater" in a tie
    assert ifn_high_accuracy > ifn_low_accuracy
    assert not (ifn_low_accuracy > ifn_high_accuracy)

    # Test 3: Equal score and equal accuracy should not be greater/less than
    ifn_clone = IFN(0.6, 0.3)
    assert not (ifn_high_accuracy > ifn_clone)
    assert not (ifn_high_accuracy < ifn_clone)
    # They should be equal, which is tested by the __eq__ method
    assert ifn_high_accuracy == ifn_clone

def test_ifn_identities():
    """Test neutral and multiplicative identity elements."""
    assert IFN.neutral_element() == IFN(0.0, 1.0)
    assert IFN.multiplicative_identity() == IFN(1.0, 0.0)

def test_ifn_inverse(ifn1):
    """Test the complement/inverse of an IFN."""
    assert ifn1.inverse() == IFN(0.3, 0.6)

def test_ifn_defuzzification(ifn1):
    """Test the different defuzzification methods for IFN."""
    # ifn1 = (0.6, 0.3) -> pi = 0.1

    # Score method: 0.6 - 0.3 = 0.3
    assert ifn1.defuzzify(method='score') == pytest.approx(0.3)
    # Centroid method should be an alias for score
    assert ifn1.defuzzify(method='centroid') == pytest.approx(0.3)

    # Value method: mu + pi*mu = 0.6 + (0.1 * 0.6) = 0.6 + 0.06 = 0.66
    assert ifn1.defuzzify(method='value') == pytest.approx(0.66)

def test_ifn_from_crisp():
    """Test creating an IFN from a crisp value [0, 1]."""
    ifn = IFN.from_crisp(0.8)
    assert isinstance(ifn, IFN)
    assert ifn.mu == 0.8
    assert ifn.nu == pytest.approx(0.2)
    assert ifn.pi == pytest.approx(0.0)

    # Test error handling for out-of-bounds crisp value
    with pytest.raises(ValueError):
        IFN.from_crisp(1.1)
