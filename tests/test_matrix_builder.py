"""
===================================================================
Tests for the Matrix Factory Module
===================================================================

This script contains unit tests for the matrix creation and conversion
functionalities, ensuring that user input is correctly transformed into
valid, reciprocal comparison matrices.
"""

import pytest
import numpy as np

# Import the code to be tested
from multiAHPy.matrix_builder import (
    FuzzyScale,
    create_comparison_matrix,
    create_matrix_from_list,
    create_matrix_from_judgments,
    complete_matrix_from_upper_triangle
)
from multiAHPy.types import TFN, TrFN, GFN, Crisp

# --- Tests for FuzzyScale Class ---

def test_fuzzyscale_tfn_conversion():
    """Test conversion of a crisp Saaty value to a TFN."""
    # Test a standard value
    tfn = FuzzyScale.get_fuzzy_number(3, TFN, scale='linear')
    assert tfn == TFN(2, 3, 4)

    # Test reciprocal value using fractions, not negative numbers
    reciprocal_tfn = FuzzyScale.get_fuzzy_number(1/3, TFN, scale='linear')
    expected_inverse = tfn.inverse()
    assert reciprocal_tfn == expected_inverse

    # Test equal importance
    equal_tfn = FuzzyScale.get_fuzzy_number(1, TFN)
    assert equal_tfn == TFN(1, 1, 1)

def test_fuzzyscale_narrow_fuzziness():
    """Test that a lower fuzziness factor creates a narrower TFN."""
    tfn = FuzzyScale.get_fuzzy_number(5, TFN, fuzziness=0.5)
    assert tfn.l == 4.5
    assert tfn.m == 5
    assert tfn.u == 5.5

def test_fuzzyscale_trfn_conversion():
    """Test conversion of a crisp Saaty value to a TrFN."""
    trfn = FuzzyScale.get_fuzzy_number(7, TrFN, fuzziness=1.0)
    assert isinstance(trfn, TrFN)
    assert trfn.a == 6
    assert trfn.b == 6.5
    assert trfn.c == 7.5
    assert trfn.d == 8

def test_fuzzyscale_crisp_conversion():
    """Test that conversion to Crisp type works as expected."""
    crisp_num = FuzzyScale.get_fuzzy_number(9, Crisp)
    assert isinstance(crisp_num, Crisp)
    assert crisp_num.value == 9

def test_fuzzyscale_invalid_input():
    """Test that invalid Saaty values raise a ValueError."""
    with pytest.raises(ValueError, match="Judgment value .* must correspond to a Saaty scale value of 1-9"):
        FuzzyScale.get_fuzzy_number(10, TFN) # Value > 9

    with pytest.raises(ValueError, match="Judgment value .* must correspond to a Saaty scale value of 1-9"):
        FuzzyScale.get_fuzzy_number(0, TFN) # Value is 0

# --- Tests for Matrix Creation Functions ---

def test_create_matrix_from_list():
    """Test creating a 3x3 TFN matrix from a flat list of judgments."""
    # C1/C2=3, C1/C3=5, C2/C3=1/2 (i.e., C3 is moderately more important than C2)
    judgments = [3, 5, 1/2]
    matrix = create_matrix_from_list(judgments, TFN, scale='linear')

    # Check shape and type
    assert matrix.shape == (3, 3)
    assert isinstance(matrix[0, 1], TFN)

    # Check diagonal
    assert matrix[0, 0] == TFN(1, 1, 1)

    # Check a value and its reciprocal
    assert matrix[0, 1] == FuzzyScale.get_fuzzy_number(3, TFN, scale='linear')
    assert matrix[1, 0] == FuzzyScale.get_fuzzy_number(1/3, TFN, scale='linear')
    assert matrix[2, 1] == FuzzyScale.get_fuzzy_number(2, TFN, scale='linear')
    
def test_create_matrix_from_list_invalid_length():
    """Test that a list with an invalid number of judgments raises an error."""
    invalid_judgments = [3, 5, 2, 4] # 4 judgments cannot form a square matrix
    with pytest.raises(ValueError, match="Invalid number of judgments"):
        create_matrix_from_list(invalid_judgments, TFN)

def test_create_matrix_from_judgments_dict():
    """Test creating a 4x4 Crisp matrix from a dictionary of judgments."""
    items = ["Price", "Safety", "Style", "Fuel Eco"]
    judgments = {
        ("Price", "Safety"): 4,
        ("Price", "Style"): 2,
        ("Safety", "Fuel Eco"): 3,
        ("Style", "Fuel Eco"): 5
    }

    matrix = create_matrix_from_judgments(judgments, items, Crisp)

    # Check shape and type
    assert matrix.shape == (4, 4)
    assert isinstance(matrix[0, 1], Crisp)

    # Check a value that was provided
    assert matrix[0, 1].value == 4
    # Check its reciprocal was set
    assert matrix[1, 0].value == 0.25

    # Check a value that was NOT provided (should be identity TFN)
    # The pair ("Price", "Fuel Eco") was not judged.
    # Note: `complete_matrix_from_upper_triangle` does not fill missing judgments,
    # it only fills reciprocals. This is expected behavior.
    assert matrix[0, 3].value == 1 # It should default to 1

def test_complete_matrix_from_upper_triangle():
    """Test that the reciprocity function correctly fills a matrix."""
    n = 3
    matrix = create_comparison_matrix(n, TFN)

    matrix[0, 1] = TFN(2, 3, 4)
    matrix[0, 2] = TFN(4, 5, 6)
    matrix[1, 2] = TFN(1, 2, 3)

    # At this point, lower triangle is still identity TFNs
    assert matrix[1, 0] != TFN(2, 3, 4).inverse()

    completed_matrix = complete_matrix_from_upper_triangle(matrix)

    # Now check if the lower triangle is correctly filled
    assert completed_matrix[1, 0] == TFN(2, 3, 4).inverse()
    assert completed_matrix[2, 0] == TFN(4, 5, 6).inverse()
    assert completed_matrix[2, 1] == TFN(1, 2, 3).inverse()

    # Check that the diagonal remains unchanged
    assert completed_matrix[1, 1] == TFN(1, 1, 1)
