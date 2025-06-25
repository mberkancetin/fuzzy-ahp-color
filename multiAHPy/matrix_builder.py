from __future__ import annotations
from typing import List, Dict, Type, TYPE_CHECKING
import numpy as np
import math

if TYPE_CHECKING:
    from multiAHPy.types import NumericType, Number, TFN, IT2TrFN, TrFN, IFN, Crisp

# ==============================================================================
# 1. FUZZY SCALE CONVERSION
# ==============================================================================

class FuzzyScale:
    """
    Handles the conversion of crisp Saaty-scale judgments (1-9) into various
    fuzzy number types using different predefined conversion scales.
    """
    _SCALES = {
        # A common, simple scale with a consistent +/- 1 spread (except at boundaries).
        # See, e.g., Mikhailov, L. (2004) for a similar linear approach.
        "linear": {
            1: (1, 1, 1), 2: (1, 2, 3), 3: (2, 3, 4), 4: (3, 4, 5),
            5: (4, 5, 6), 6: (5, 6, 7), 7: (6, 7, 8), 8: (7, 8, 9), 9: (8, 9, 9)
        },
        # A scale often cited, e.g., in work by Ayhan & Kilic (2015). The value 9
        # is often treated as crisp to represent absolute certainty.
        "saaty_original": {
            1: (1, 1, 1), 2: (1, 2, 3), 3: (2, 3, 4), 4: (3, 4, 5),
            5: (4, 5, 6), 6: (5, 6, 7), 7: (6, 7, 8), 8: (7, 8, 9), 9: (9, 9, 9)
        },
        # A scale with a wider spread, representing higher uncertainty in judgments
        "wide": {
            1: (1, 1, 3), 2: (1, 2, 4), 3: (2, 3, 5), 4: (3, 4, 6),
            5: (4, 5, 7), 6: (5, 6, 8), 7: (6, 7, 9), 8: (7, 8, 9), 9: (8, 9, 9)
        },
        # A scale with a very small, uniform spread, representing high confidence.
        "narrow": {
            1: (1, 1, 1), 2: (1.5, 2, 2.5), 3: (2.5, 3, 3.5), 4: (3.5, 4, 4.5),
            5: (4.5, 5, 5.5), 6: (5.5, 6, 6.5), 7: (6.5, 7, 7.5), 8: (7.5, 8, 8.5), 9: (8.5, 9, 9)
        }
    }

    _IFN_SCALE = {
        # Based on Nguyen (2019), "A new method...using intuitionistic fuzzy numbers"
        # (mu, nu) pairs for linguistic terms 1 through 9
        1: (0.50, 0.40), 2: (0.55, 0.35), 3: (0.60, 0.30), 4: (0.65, 0.25),
        5: (0.70, 0.20), 6: (0.75, 0.15), 7: (0.80, 0.10), 8: (0.90, 0.05), 9: (1.00, 0.00)
    }

    @staticmethod
    def available_scales() -> List[str]:
        """Returns a list of available scale types."""
        return list(FuzzyScale._SCALES.keys())

    @staticmethod
    def get_fuzzy_number(
        crisp_value: int,
        number_type: Type[Number],
        scale: str = 'linear',
        fuzziness: float = None,
        umf_spread: float = 1.5,
        lmf_spread: float = 0.5
    ) -> Number:
        """
        Converts a crisp judgment (1-9) to a fuzzy number using a named scale.

        Args:
            crisp_value: The Saaty scale integer (1-9 or -9 to -1 for reciprocals).
            number_type: The target fuzzy number class (e.g., TFN, Crisp).
            scale: The named scale to use ('linear', 'saaty_original', 'wide', etc.).
            fuzziness: GFN class only attribute.
            umf_spread: IT2TrFN class only attribute to create a wider UMF
            lmf_spread: IT2TrFN class only attribute to create a narrower LMF

        Returns:
            An instance of the specified fuzzy number type.
        """
        if scale not in FuzzyScale._SCALES:
            raise ValueError(f"Unknown scale: '{scale}'. Available scales: {FuzzyScale.available_scales()}")

        if not isinstance(crisp_value, (int, float)):
            raise TypeError("Crisp judgment must be a number.")

        is_reciprocal = False
        value = crisp_value

        if 0 < abs(value) < 1:
            is_reciprocal = True
            value = 1 / value

        # Round the value to the nearest integer to use it as a key for our scales.
        # This handles cases like 1/3 (0.333...) whose reciprocal is 3.
        # It also handles a direct input of 3.1 being treated as 3.
        value = int(round(value))

        if not (1 <= value <= 9):
            raise ValueError(f"Judgment value ({crisp_value}) must correspond to a Saaty scale value of 1-9.")

        type_name = number_type.__name__

        # Base case: Equal importance
        if value == 1:
            return number_type.multiplicative_identity()

        # Define standard spreads for TFN and TrFN based on fuzziness
        spread = fuzziness
        if type_name == 'TFN':
            if spread:
                params = (max(1, value - spread), value, value + spread)
            else:
                params = FuzzyScale._SCALES[scale][value]
        elif type_name == 'TrFN':
            params = (max(1, value - spread), value - spread/2, value + spread/2, value + spread)
        elif type_name == 'GFN':
            params = (value, value * (fuzziness / 10.0))
        elif type_name == 'IFN':
            params = FuzzyScale._IFN_SCALE[value]
        elif type_name == 'IT2TrFN':
            l, m, u = FuzzyScale._SCALES[scale][abs(crisp_value)]

            umf = TrFN.from_tfn(TFN(max(1, m-umf_spread), m, m+umf_spread))
            lmf = TrFN.from_tfn(TFN(m-lmf_spread, m, m+lmf_spread))

            it2_num = IT2TrFN(umf=umf, lmf=lmf)
            return it2_num.inverse() if crisp_value < 0 else it2_num
        elif type_name == 'Crisp':
            return number_type(crisp_value)
        else:
            raise TypeError(f"Unsupported number_type for fuzzy scaling: {type_name}")

        fuzzy_num = number_type(*params)
        return fuzzy_num.inverse() if is_reciprocal else fuzzy_num

# ==============================================================================
# 2. MATRIX CREATION AND MANIPULATION
# ==============================================================================

def _get_matrix_size_from_list_len(num_judgments: int) -> int:
    """
    Calculates the size 'n' of a square matrix given 'k' pairwise judgments
    from its upper triangle. Solves the equation n*(n-1)/2 = k.

    Returns:
        The integer size 'n' of the matrix.

    Raises:
        ValueError: If the number of judgments does not correspond to a valid matrix.
    """
    # The formula for the number of pairs k in an n x n matrix is k = n*(n-1)/2
    # Rearranging this gives a quadratic equation: n^2 - n - 2k = 0
    # We solve for n using the quadratic formula: n = (-b ± sqrt(b^2 - 4ac)) / 2a
    # Here, a=1, b=-1, c=-2k

    discriminant = 1 - (4 * 1 * (-2 * num_judgments))
    if discriminant < 0:
        raise ValueError(f"Invalid number of judgments ({num_judgments}). Cannot form a square matrix.")

    # We only care about the positive root
    n = (1 + math.sqrt(discriminant)) / 2

    # Check if n is an integer
    if n != int(n):
        raise ValueError(f"Invalid number of judgments ({num_judgments}). Does not correspond to a full upper-triangle matrix.")

    return int(n)

def create_matrix_from_list(
    judgments: List[int | float],
    number_type: Type[Number],
    scale: str = 'linear',
    fuzziness: float = None
) -> np.ndarray:
    """
    Creates a complete, reciprocal comparison matrix from a flattened list of
    upper-triangle judgments (read row by row).

    Example: For a 4x4 matrix, the list should contain 6 judgments for the
    pairs (1,2), (1,3), (1,4), (2,3), (2,4), (3,4) in that order.

    [C1/C2, C1/C3, C1/C4, C2/C3, C2/C4, C3/C4]

    Args:
        judgments: A flat list of crisp judgment values (1-9).
        number_type: The target number class (e.g., Crisp, TFN).
        scale: The named scale to use ('linear', 'saaty_original', 'wide', etc.).
        fuzziness: The fuzziness factor for fuzzy conversions.

    Returns:
        A complete, reciprocal comparison matrix.
    """
    num_judgments = len(judgments)
    size = _get_matrix_size_from_list_len(num_judgments)

    matrix = create_comparison_matrix(size, number_type)

    # Use an iterator for the judgments list for easy consumption
    judgment_iterator = iter(judgments)

    # Fill the upper triangle of the matrix
    for i in range(size):
        for j in range(i + 1, size):
            try:
                crisp_value = next(judgment_iterator)
                matrix[i, j] = FuzzyScale.get_fuzzy_number(crisp_value, number_type, fuzziness=fuzziness, scale=scale)
            except StopIteration:
                # This should not happen if _get_matrix_size_from_list_len is correct
                raise ValueError("Mismatch between number of judgments and calculated matrix size.")

    # Fill the lower triangle with reciprocals
    return complete_matrix_from_upper_triangle(matrix)

def create_comparison_matrix(size: int, number_type: Type[Number]) -> np.ndarray:
    """
    Creates an (n x n) pairwise comparison matrix where each element
    is an object of the specified number_type, initialized to the
    multiplicative identity (e.g., 1).
    """
    matrix = np.empty((size, size), dtype=object)
    identity = number_type.multiplicative_identity()
    for i in range(size):
        for j in range(size):
            matrix[i, j] = identity

    return matrix

def complete_matrix_from_upper_triangle(matrix: np.ndarray, consistency_method: str = "centroid") -> np.ndarray:
    """Fills the lower triangle using reciprocals of the upper triangle."""
    n = matrix.shape[0]
    completed_matrix = matrix.copy()
    identity_val = matrix[0,0].multiplicative_identity().defuzzify(method=consistency_method)

    for i in range(n):
        for j in range(i + 1, n):
            # Check if the upper-triangle element has been changed from its default identity value
            if abs(completed_matrix[i, j].defuzzify(method=consistency_method) - identity_val) > 1e-9:
                completed_matrix[j, i] = completed_matrix[i, j].inverse()
            elif abs(completed_matrix[j, i].defuzzify(method=consistency_method) - identity_val) > 1e-9:
                 completed_matrix[i, j] = completed_matrix[j, i].inverse()
    return completed_matrix

def create_matrix_from_judgments(
    judgments: Dict[tuple, int],
    items: List[str],
    number_type: Type[Number],
    scale: str = 'linear',
    fuzziness: float = None
) -> np.ndarray:
    """
    Creates a complete comparison matrix from a dictionary of crisp judgments,
    using a specified fuzzy conversion scale.
    """
    n = len(items)
    item_map = {name: i for i, name in enumerate(items)}
    matrix = create_comparison_matrix(n, number_type)

    for (item1, item2), value in judgments.items():
        try:
            i, j = item_map[item1], item_map[item2]
        except KeyError as e:
            raise ValueError(f"Item '{e.args[0]}' in judgments not found in the list of items.") from e

        # Ensure judgments are for the upper triangle (i < j)
        if i >= j:
            raise ValueError(f"Judgment '{item1}' vs '{item2}' is not in the upper triangle. Please only provide one judgment per pair.")

        matrix[i, j] = FuzzyScale.get_fuzzy_number(value, number_type, fuzziness=fuzziness, scale=scale)

    return complete_matrix_from_upper_triangle(matrix)
