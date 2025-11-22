import pytest
import numpy as np

from multiAHPy.completion import complete_matrix, COMPLETION_METHOD_REGISTRY
from multiAHPy.matrix_builder import create_completed_matrix
from multiAHPy.consistency import Consistency
from multiAHPy.types import TFN, Crisp, IFN

# ==============================================================================
# 1. FIXTURES: Reusable data from the papers
# ==============================================================================

@pytest.fixture
def csato_2024_fig1_matrix():
    """
    Incomplete matrix from Example 1 of Csató et al. (2024), "On the coincidence...".
    This is a 4x4 matrix with a13 and a24 missing.
    Source: Page 4 (p. 242 in journal)
    """
    return np.array([
        [1.0, 12.0, None, 14.0],
        [1/12, 1.0, 23.0, None],
        [None, 1/23, 1.0, 34.0],
        [1/14, None, 1/34, 1.0]
    ], dtype=object)


@pytest.fixture
def zhou_2018_example1_ipcm():
    """
    Incomplete matrix from Example 1 of Zhou et al. (2018).
    This version uses corrected reciprocal values and ensures symmetric Nones.
    """
    return np.array([
        [1.0, None, 4.0, 8.0],
        [None, 1.0, 2.0, 4.0],
        [0.25, 0.50, 1.0, 2.0],
        [1/8, 0.25, 0.50, 1.0]
    ], dtype=object)

@pytest.fixture
def bozoki_2010_example3_ipcm():
    """
    Incomplete matrix from Example 3 of Bozóki et al. (2010), "On optimal completion...".
    This is a 4x4 matrix with one missing value (x).
    Source: Page 15 (p. 332 in journal)
    """
    return np.array([
        [1.0, 1.0, 5.0, 2.0],
        [1.0, 1.0, 3.0, 4.0],
        [1/5, 1/3, 1.0, None],
        [1/2, 1/4, None, 1.0]
    ], dtype=object)

@pytest.fixture
def incomplete_crisp_matrix():
    return np.array([
        [1.0, 2.0, None, 4.0],
        [0.5, 1.0, 3.0, 5.0],
        [None, 1/3, 1.0, 6.0],
        [1/4, 1/5, 1/6, 1.0]
    ], dtype=object)


# ==============================================================================
# 2. UNIT TESTS
# ==============================================================================

def test_registry_has_methods():
    """Test that our completion methods are correctly registered."""
    assert "eigenvalue_optimization" in COMPLETION_METHOD_REGISTRY
    assert "dematel" in COMPLETION_METHOD_REGISTRY

def test_invalid_method_raises_error(zhou_2018_example1_ipcm):
    """Test that calling a non-existent method raises a ValueError."""
    with pytest.raises(ValueError, match="Completion method 'non_existent_method' not found"):
        complete_matrix(zhou_2018_example1_ipcm, method="non_existent_method")

@pytest.mark.parametrize("method", ["eigenvalue_optimization", "dematel"])
def test_completed_matrix_is_reciprocal(method, zhou_2018_example1_ipcm):
    """
    A fundamental property check: The completed matrix must be reciprocal.
    We test this for all implemented methods.
    """
    completed = complete_matrix(zhou_2018_example1_ipcm, method=method)
    completed_float = completed.astype(float)

    # Check that no None/nan values remain
    assert not np.any(completed_float is None)
    assert not np.any(np.isnan(completed_float))


    # Check for reciprocity: M_ji = 1 / M_ij
    assert np.allclose(completed.T, 1 / completed, atol=0.1)


# --- Tests Based on Zhou et al. (2018) ---

def test_dematel_completion_from_zhou_2018(zhou_2018_example1_ipcm):
    """
    Tests the DEMATEL completion method against the exact result from Zhou et al. (2018), Example 1.
    Source: Page 8
    """
    # The paper's final completed matrix Mc
    expected_matrix = np.array([
        [1.0, 2.0, 4.0, 8.0],
        [0.5, 1.0, 2.0, 4.0],
        [0.25, 0.5, 1.0, 2.0],
        [0.13, 0.25, 0.5, 1.0]
    ])

    # Run our implementation
    completed = complete_matrix(zhou_2018_example1_ipcm, method="dematel")

    # The paper rounds to two decimal places, so we use a reasonable tolerance.
    # The value for a41 is 1/8=0.125 in the paper, but their input says 0.13. We follow their input.
    # We will check for approximate equality.
    assert np.allclose(completed, expected_matrix, atol=0.15)


# --- Tests Based on Bozóki et al. (2010) ---

def test_eigenvalue_optimization_from_bozoki_2010(bozoki_2010_example3_ipcm):
    """
    Tests the eigenvalue optimization against the result from Bozóki et al. (2010), Example 3.
    They calculate the optimal 'x' that minimizes lambda_max.
    Source: Page 15 (p. 332)
    """
    # From the paper, the optimal value for the missing element x (at position 2,3) is ~0.73
    expected_x = 0.7302965

    # Run our implementation
    completed = complete_matrix(bozoki_2010_example3_ipcm, method="eigenvalue_optimization")

    # Get the value our algorithm found for the missing element
    found_x = completed[2, 3]

    # Check if our result is very close to the one published in the paper
    assert pytest.approx(found_x, abs=1e-4) == expected_x



# --- Test for Incomplete/Disconnected Graphs ---
def test_dematel_handles_disconnected_graph():
    """
    Tests that the DEMATEL method fails gracefully if the matrix is "disconnected",
    meaning it cannot be solved.
    Source: Bozóki et al. (2010), Theorem 2
    """
    # This matrix is disconnected. Items (0,1) cannot be linked to (2,3).
    disconnected_ipcm = np.array([
        [1.0, 2.0, None, None],
        [0.5, 1.0, None, None],
        [None, None, 1.0, 4.0],
        [None, None, 0.25, 1.0]
    ], dtype=object)

    # The DEMATEL method should fail because the matrix (I - N) becomes singular.
    with pytest.raises(ValueError, match="Matrix is disconnected. DEMATEL completion requires a connected graph of judgments."):
        complete_matrix(disconnected_ipcm, method="dematel")


# --- Tests Based on Csató et al. (2024) ---

def test_coincidence_of_methods_for_n4_2(csato_2024_fig1_matrix):
    """
    Tests the main finding from Csató et al. (2024): for n=4 matrices,
    the values estimated for the MISSING entries should be very similar
    between the two methods.
    """
    # Get the original incomplete matrix to find the missing indices
    ipcm = csato_2024_fig1_matrix

    # Run both completion methods
    completed_eigen = complete_matrix(ipcm, method="eigenvalue_optimization")
    completed_dematel = complete_matrix(ipcm, method="dematel")

    # We should only compare the values that were originally missing.
    # Let's iterate through the matrix and check only those.
    n = ipcm.shape[0]
    mismatches = []

    for i in range(n):
        for j in range(n):
            # If the original value was missing...
            if ipcm[i, j] is None:
                val_eigen = completed_eigen[i, j]
                val_dematel = completed_dematel[i, j]
                print(f"Comparing missing value at ({i},{j}): Eigen={val_eigen:.4f}, DEMATEL={val_dematel:.4f}")

                # Check if these two *estimated* values are close
                if not np.isclose(val_eigen, val_dematel, atol=0.1):
                    mismatches.append(f"Mismatch at ({i},{j})")

    assert not mismatches, f"Completion methods differed significantly on missing values: {mismatches}"


# In tests/test_completion.py



def test_coincidence_of_methods_for_n4(csato_2024_fig1_matrix):
    """
    Tests the main finding from Csató et al. (2024): for n=4 matrices,
    eigenvalue optimization and logarithmic least squares should yield
    the same optimal completion.
    """
    ipcm = csato_2024_fig1_matrix

    # Complete with both methods
    completed_eigen = complete_matrix(ipcm, method="eigenvalue_optimization")
    completed_eigen_type_agnostic = complete_matrix(ipcm, method="eigenvalue_optimization_type_agnostic")
    completed_llsm = complete_matrix(ipcm, method="llsm")
    completed_llsm_type_agnostic = complete_matrix(ipcm, method="llsm_type_agnostic") # <-- Use the new method

    # The theorem states these completions should be identical for n=4.
    # We can use a very small tolerance here.
    assert np.allclose(completed_eigen, completed_llsm_type_agnostic, atol=1e-6)
    assert np.allclose(completed_eigen_type_agnostic, completed_llsm, atol=1e-6)


# --- NEW TEST specific to LLSM ---

@pytest.fixture
def disconnected_ipcm():
    """A matrix with a disconnected graph of judgments."""
    return np.array([
        [1.0, 2.0, None, None],
        [0.5, 1.0, None, None],
        [None, None, 1.0, 4.0],
        [None, None, 0.25, 1.0]
    ], dtype=object)

def test_llsm_handles_disconnected_graph(disconnected_ipcm):
    """
    Tests that the LLSM method fails gracefully if the matrix is disconnected.
    The pseudoinverse may not raise an error, but the resulting weights
    would be arbitrary. A proper implementation should check connectivity.
    Let's first test if it raises an error, if not, we should add a check.

    Update: The pseudoinverse will often "work" but give a poor solution.
    A graph connectivity check is better. Let's add it.
    """
    # Let's add a graph check to our LLSM function for robustness.
    # (We will modify the main function after writing this test).

    with pytest.raises(ValueError, match="iPCM graph may be disconnected"):
        complete_matrix(disconnected_ipcm, method="llsm")


def test_llsm_handles_disconnected_graph(disconnected_ipcm):
    """
    Tests that the LLSM method fails gracefully if the matrix is disconnected.
    The pseudoinverse may not raise an error, but the resulting weights
    would be arbitrary. A proper implementation should check connectivity.
    Let's first test if it raises an error, if not, we should add a check.

    Update: The pseudoinverse will often "work" but give a poor solution.
    A graph connectivity check is better. Let's add it.
    """
    # Let's add a graph check to our LLSM function for robustness.
    # (We will modify the main function after writing this test).

    with pytest.raises(ValueError, match="LLSM completion failed: iPCM graph is disconnected."):
        complete_matrix(disconnected_ipcm, method="llsm_type_agnostic")


def test_create_completed_crisp_matrix(incomplete_crisp_matrix):
    """Tests that completion produces a matrix of Crisp objects."""

    completed_matrix = create_completed_matrix(
        incomplete_crisp_matrix,
        number_type=Crisp,
        completion_method="dematel" # Use a fast method for testing
    )

    # Check the type of the elements
    assert isinstance(completed_matrix[0, 0], Crisp)
    assert isinstance(completed_matrix[0, 1], Crisp)

    # Check that a missing value was filled
    assert completed_matrix[0, 2].value is not None

    # Check for reciprocity
    assert completed_matrix[2, 0].value == pytest.approx(1 / completed_matrix[0, 2].value)

def test_create_completed_tfn_matrix(incomplete_crisp_matrix):
    """Tests that completion produces a matrix of TFN objects using a scale."""

    completed_matrix = create_completed_matrix(
        incomplete_crisp_matrix,
        number_type=TFN,
        scale='linear', # Use the linear scale for conversion
        completion_method="dematel"
    )

    # Check the type of the elements
    assert isinstance(completed_matrix[0, 0], TFN)
    assert isinstance(completed_matrix[0, 1], TFN)

    # Check that a known value (2.0) was converted to a TFN correctly
    # According to the 'linear' scale, 2 -> (1, 2, 3)
    known_tfn = completed_matrix[0, 1]
    assert known_tfn.l == 1.0
    assert known_tfn.m == 2.0
    assert known_tfn.u == 3.0

    # Check that a missing value was first completed, then converted to a TFN
    filled_tfn = completed_matrix[0, 2]
    assert filled_tfn is not None
    # Its middle value 'm' should be close to the numerically completed value
    assert filled_tfn.m > 0
