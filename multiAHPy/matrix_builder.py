from __future__ import annotations
from typing import List, Dict, Type, TYPE_CHECKING
import numpy as np
import math
from .config import configure_parameters
from .completion import complete_matrix
from .weight_derivation import derive_weights
from .types import Number, Crisp


if TYPE_CHECKING:
    from .types import NumericType, Number, TFN, IT2TrFN, TrFN, IFN, Crisp

# ==============================================================================
# 1. FUZZY SCALE CONVERSION
# ==============================================================================

class FuzzyScale:
    """
    Handles the conversion of crisp Saaty-scale judgments (1-9) into various
    fuzzy number types using different predefined conversion scales.
    """
    @staticmethod
    def available_tfn_scales() -> List[str]:
        """Returns a list of available TFN scale types from the global config."""
        return list(configure_parameters.FUZZY_TFN_SCALES_FUNCTIONS.keys())

    @staticmethod
    def available_ifn_scales() -> List[str]:
        """Returns a list of available TFN scale types from the global config."""
        return list(configure_parameters.FUZZY_IFN_SCALES.keys())

    @staticmethod
    def get_fuzzy_number(
        crisp_value: float,
        number_type: Type[Number],
        scale: str = 'linear',
        fuzziness: float = None,
        umf_spread: float = 1.5,
        lmf_spread: float = 0.5
    ) -> Number:
        """
        Converts a crisp judgment value to a fuzzy number using a registered scale function.
        This method now handles any float value for TFN/TrFN scales.

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
        if scale not in configure_parameters.FUZZY_TFN_SCALES_FUNCTIONS:
            if scale not in configure_parameters.FUZZY_IFN_SCALES_FUNCTIONS:
                raise ValueError(f"Unknown scale: '{scale}'. Available scales: {FuzzyScale.available_tfn_scales()}")

        if not isinstance(crisp_value, (int, float, np.number)):
            raise TypeError("Crisp judgment must be a number.")

        type_name = number_type.__name__

        if type_name == 'Crisp':
            return number_type(crisp_value)

        is_reciprocal = False
        value = float(crisp_value)

        # Handle logic for reciprocals
        if 0 < abs(value) < 1:
            is_reciprocal = True
            value = 1.0 / value

        if np.isclose(value, 1.0):
            if type_name == 'TFN':
                return number_type(1.0, 1.0, 1.0)
            elif type_name == 'TrFN':
                return number_type(1.0, 1.0, 1.0, 1.0)

        spread = fuzziness
        params = None

        if type_name == 'TFN':
            if spread:
                params = (max(1, value - spread), value, value + spread)
            else:
                scale_func = configure_parameters.FUZZY_TFN_SCALES_FUNCTIONS.get(scale)
                params = scale_func(value)

        elif type_name == 'TrFN':
            if spread:
                params = (max(1, value - spread), value - spread/2, value + spread/2, value + spread)
            else:
                scale_func = configure_parameters.FUZZY_TFN_SCALES_FUNCTIONS.get(scale)
                params = scale_func(value)
                if len(params) == 3: # Handle TFN scale returned for TrFN request
                    params = (params[0], params[1], params[1], params[2])

        elif type_name == 'GFN':
            key = int(np.clip(round(value), 1, 9))
            params = (key, key * (fuzziness / 10.0))

        elif type_name == 'IFN':
            scale_func = configure_parameters.FUZZY_IFN_SCALES_FUNCTIONS.get(scale)
            if scale_func is None:
                if scale not in configure_parameters.FUZZY_IFN_SCALES_FUNCTIONS:
                     print(f"Warning: IFN scale '{scale}' not found. Defaulting to 'buyukozkan_9_level'.")
                     scale_func = configure_parameters.FUZZY_IFN_SCALES_FUNCTIONS['buyukozkan_9_level']
            params = scale_func(value)

        elif type_name == 'IT2TrFN':
            l, m, u = configure_parameters.FUZZY_TFN_SCALES_FUNCTIONS[scale][abs(crisp_value)]
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
    # The formula for the number of pairs k in an n x n matrix is
    # k = n*(n-1)/2

    # Rearranging this gives a quadratic equation:
    # n^2 - n - 2k = 0

    # We solve for n using the quadratic formula:
    # n = (-b ± sqrt(b^2 - 4ac)) / 2a

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

    Special Logic for IFN:
    Creates a full Crisp matrix first, ensuring A_ji = 1/A_ij (arithmetic reciprocal),
    and THEN maps every element to IFN. This guarantees that if A_ij = 1, A_ji = 1,
    and both map to the same IFN (e.g. 0.5, 0.4), avoiding positional bias.
    """
    # --- IFN Specific Logic: Crisp First Strategy ---
    if number_type.__name__ == 'IFN':
        # 1. Create a complete, reciprocally consistent Crisp matrix
        crisp_matrix = create_matrix_from_list(judgments, Crisp, scale=scale)

        # 2. Convert every cell to IFN using the scale
        n = crisp_matrix.shape[0]
        ifn_matrix = create_comparison_matrix(n, number_type)

        for i in range(n):
            for j in range(n):
                val = crisp_matrix[i, j].value
                ifn_matrix[i, j] = FuzzyScale.get_fuzzy_number(val, number_type, scale=scale)

        return ifn_matrix

    # --- Standard Logic for TFN/TrFN/Crisp ---
    num_judgments = len(judgments)
    size = _get_matrix_size_from_list_len(num_judgments)
    matrix = create_comparison_matrix(size, number_type)
    judgment_iterator = iter(judgments)

    for i in range(size):
        for j in range(i + 1, size):
            try:
                crisp_value = next(judgment_iterator)
                matrix[i, j] = FuzzyScale.get_fuzzy_number(crisp_value, number_type, fuzziness=fuzziness, scale=scale)
            except StopIteration:
                raise ValueError("Mismatch between number of judgments and calculated matrix size.")

    return complete_matrix_from_upper_triangle(matrix)

def create_comparison_matrix(size: int, number_type: Type[Number]) -> np.ndarray:
    """
    Creates an (n x n) pairwise comparison matrix where each element
    is an object of the specified number_type, initialized to the
    multiplicative identity (e.g., 1).
    """
    matrix = np.empty((size, size), dtype=object)

    # Use from_saaty(1.0) instead of multiplicative_identity().
    # For IFN, multiplicative_identity is (1,0) (Absolute Truth), but AHP diagonal is Equal (0.45, 0.45).
    # For TFN, both are (1,1,1).
    ahp_identity = number_type.from_saaty(1.0)

    for i in range(size):
        for j in range(size):
            matrix[i, j] = ahp_identity

    return matrix

def complete_matrix_from_upper_triangle(matrix: np.ndarray, consistency_method: str = "centroid", tolerance: float | None = None) -> np.ndarray:
    """
    Ensures the matrix is reciprocal and has correctly defined identity elements on the diagonal.

    Logic:
    1. Sets the diagonal [i,i] to the exact AHP "Equal Importance" value (Saaty 1).
    2. Iterates through the upper triangle [i,j].
    3. If [i,j] contains user data (differs from identity), fill [j,i] with inverse.
    4. If [i,j] is default/None but [j,i] has user data, fill [i,j] with inverse.
    5. If both are default/None, reset both to strict identity.

    Args:
        matrix: The input comparison matrix.
        consistency_method: Defuzzification method used to check if a value is "identity".
        tolerance: Tolerance for the identity check.
    """
    from .config import configure_parameters
    final_tolerance = tolerance if tolerance is not None else configure_parameters.FLOAT_TOLERANCE

    n = matrix.shape[0]
    completed_matrix = matrix.copy()

    # 1. Determine the AHP Identity Element (Saaty 1.0)
    # Try to deduce type from matrix content
    elem = completed_matrix[0, 0]
    if elem is None:
         # Search for any non-None element to determine type
         for row in completed_matrix:
             for cell in row:
                 if cell is not None:
                     elem = cell
                     break
             if elem: break

    if elem is None: raise ValueError("Matrix contains only None values; cannot determine Number type.")

    number_type = type(elem)

    # Get the strict object for "Equal Importance"
    # For TFN: (1,1,1). For IFN: (0.45, 0.45) or (0.5, 0.4) depending on scale implementation.
    ahp_identity = number_type.from_saaty(1.0)

    # Get crisp value for comparison logic
    try:
        identity_crisp = ahp_identity.defuzzify(method=consistency_method)
    except:
        identity_crisp = 1.0 # Fallback

    for i in range(n):
        # Enforce Diagonal
        completed_matrix[i, i] = ahp_identity

        for j in range(i + 1, n):
            upper_val = completed_matrix[i, j]
            lower_val = completed_matrix[j, i]

            upper_is_set = upper_val is not None
            lower_is_set = lower_val is not None

            # Function to check if a value is effectively "Equal Importance" (Default)
            def is_identity(val):
                try:
                    return abs(val.defuzzify(method=consistency_method) - identity_crisp) <= final_tolerance
                except:
                    return False

            upper_is_identity = upper_is_set and is_identity(upper_val)
            lower_is_identity = lower_is_set and is_identity(lower_val)

            # Logic: If one side is "informative" (non-identity user input), drive the other.

            if upper_is_set and not upper_is_identity:
                # Standard case: Upper is user input -> Overwrite Lower
                completed_matrix[j, i] = upper_val.inverse()

            elif lower_is_set and not lower_is_identity:
                # Reverse case: Lower is user input -> Overwrite Upper
                completed_matrix[i, j] = lower_val.inverse()

            else:
                # Both are identity or None. Force strict AHP identity on both.
                completed_matrix[i, j] = ahp_identity
                completed_matrix[j, i] = ahp_identity.inverse()

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

    Special Logic for IFN: Crisp First Strategy (see create_matrix_from_list).
    """
    # --- IFN Specific Logic ---
    if number_type.__name__ == 'IFN':
        # 1. Create complete Crisp matrix
        crisp_matrix = create_matrix_from_judgments(judgments, items, Crisp, scale=scale)

        # 2. Convert to IFN
        n = crisp_matrix.shape[0]
        ifn_matrix = create_comparison_matrix(n, number_type)

        for i in range(n):
            for j in range(n):
                val = crisp_matrix[i, j].value
                ifn_matrix[i, j] = FuzzyScale.get_fuzzy_number(val, number_type, scale=scale)
        return ifn_matrix

    # --- Standard Logic ---
    n = len(items)
    item_map = {name: i for i, name in enumerate(items)}
    matrix = create_comparison_matrix(n, number_type)

    for (item1, item2), value in judgments.items():
        try:
            i, j = item_map[item1], item_map[item2]
        except KeyError as e:
            raise ValueError(f"Item '{e.args[0]}' in judgments not found in the list of items.") from e

        if i >= j:
            raise ValueError(f"Judgment '{item1}' vs '{item2}' is not in the upper triangle.")

        matrix[i, j] = FuzzyScale.get_fuzzy_number(value, number_type, fuzziness=fuzziness, scale=scale)

    return complete_matrix_from_upper_triangle(matrix)

def create_completed_matrix(
    incomplete_matrix: np.ndarray,
    number_type: Type[Number],
    scale: str = 'linear',
    completion_method: str = "eigenvalue_optimization",
    fuzziness: float = None,
    **kwargs
) -> np.ndarray:
    """
    Creates a complete, type-aware comparison matrix from an incomplete one.

    This function serves as a high-level wrapper. It first uses a numerical
    algorithm to fill in the missing values and then converts the resulting
    crisp matrix into a matrix of the desired fuzzy number type using the
    specified scale.

    Args:
        incomplete_matrix: A square NumPy array (dtype=object) with missing
                           values represented by `None` or `np.nan`.
        number_type: The target number class (e.g., Crisp, TFN, IFN).
        scale: The named scale to use for converting the completed crisp
               judgments into fuzzy numbers (e.g., 'linear', 'wide').
        completion_method: The underlying numerical algorithm to use for
                           filling in missing values.
        **kwargs: Additional arguments to pass to the completion method
                  (e.g., max_iter, tolerance).

    Returns:
        A complete, reciprocal comparison matrix of the specified number_type.
    """
    print(f"\n--- Creating completed matrix (Type: {number_type.__name__}, Completion: {completion_method}) ---")

    # --- Step 1: Use the numerical engine to get a completed crisp matrix ---
    # We must convert the incomplete matrix to float first, replacing None with np.nan
    # so that the numerical methods can handle it.

    # Create a copy to work with, ensuring dtype is object to hold Nones
    ipcm_copy = incomplete_matrix.copy().astype(object)
    ipcm_copy[ipcm_copy == None] = np.nan

    # Numerical completion
    completed_crisp_matrix = complete_matrix(
        incomplete_matrix=ipcm_copy,
        method=completion_method,
        **kwargs
    )

    # Convert to fuzzy type
    n = completed_crisp_matrix.shape[0]
    final_fuzzy_matrix = create_comparison_matrix(n, number_type)

    for i in range(n):
        for j in range(i + 1, n):
            crisp_value = completed_crisp_matrix[i, j]
            final_fuzzy_matrix[i, j] = FuzzyScale.get_fuzzy_number(
                crisp_value,
                number_type,
                fuzziness=fuzziness,
                scale=scale
            )

    return complete_matrix_from_upper_triangle(final_fuzzy_matrix)

def create_matrix_dynamic_hesitancy(
    judgments: Dict[tuple, int] | List[float],
    n: int, # Matrix size
    consistency_method: str = "centroid"
) -> np.ndarray:
    """
    Builds an IFN matrix where hesitation is derived from the crisp consistency ratio.
    """
    from .consistency import Consistency
    from .types import IFN, Crisp

    # PASS 1: Build Temporary Crisp Matrix
    # We use your existing logic but force type to Crisp
    if isinstance(judgments, list):
        crisp_matrix = create_matrix_from_list(judgments, Crisp)
    else:
        # Assuming dict judgments provided, items list required, simplified here
        raise NotImplementedError("For dynamic scale, use list input or adapt logic")

    # PASS 2: Calculate Consistency (CR)
    # We calculate CR on the crisp proxy.
    cr = Consistency.calculate_saaty_cr(
        crisp_matrix,
        consistency_method=consistency_method
    )

    print(f"  - Dynamic Scale: Detected CR={cr:.4f}. Adjusting hesitation...")

    # PASS 3: Generate IFN Matrix
    ifn_matrix = create_comparison_matrix(n, IFN)

    for i in range(n):
        for j in range(n):
            if i == j:
                # Identity with base hesitation
                ifn_matrix[i, j] = IFN.from_saaty_with_consistency(1.0, matrix_cr=0.0)
            else:
                val = crisp_matrix[i, j].value
                # Generate IFN using the CR of the whole matrix
                ifn_matrix[i, j] = IFN.from_saaty_with_consistency(
                    val,
                    matrix_cr=cr,
                    hesitation_factor=1.0 # 1.0 means if CR is 0.1, added hesitation is 0.1
                )

    return ifn_matrix

def rebuild_consistent_matrix(inconsistent_matrix: np.ndarray) -> np.ndarray:
    """
    Creates a new, perfectly consistent matrix from an inconsistent one.

    This is achieved by:
    1. Calculating the priority vector (weights) from the inconsistent matrix
       using the robust geometric mean method.
    2. Building a new matrix where each element a_ij is the ratio w_i / w_j.

    The resulting matrix is perfectly consistent (its Saaty's CR will be 0.0).
    This is a core technique for consistency optimization.

    Args:
        inconsistent_matrix: A complete, square NumPy array of numerical judgments.

    Returns:
        A new, perfectly consistent NumPy array of the same size.
    """
    n = inconsistent_matrix.shape[0]
    try:
        matrix = inconsistent_matrix.astype(float)
    except (ValueError, TypeError):
        raise TypeError("Input must be numerical.")

    sanitized_matrix = np.maximum(matrix, 1e-9)
    try:
        log_matrix = np.log(sanitized_matrix)
        weights = np.exp(np.mean(log_matrix, axis=1))
        weights /= np.sum(weights)
    except Exception:
        weights = np.full(n, 1.0 / n)

    consistent_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if weights[j] > 1e-9:
                consistent_matrix[i, j] = weights[i] / weights[j]
            else:
                consistent_matrix[i, j] = 1.0
    return consistent_matrix

def rebuild_from_eigenvector(inconsistent_matrix: np.ndarray) -> np.ndarray:
    """
    Creates a new, perfectly consistent matrix from an inconsistent one using
    the principal right eigenvector method.

    Args:
        inconsistent_matrix: A complete, square NumPy array of numerical judgments.

    Returns:
        A new, perfectly consistent NumPy array of the same size.
    """
    n = inconsistent_matrix.shape[0]
    try:
        matrix = inconsistent_matrix.astype(float)
    except (ValueError, TypeError):
        raise TypeError("Input must be numerical.")

    crisp_object_matrix = np.array([[Crisp(c) for c in row] for row in matrix], dtype=object)
    try:
        results = derive_weights(crisp_object_matrix, Crisp, method="eigenvector")
        weights = results['crisp_weights']
    except Exception:
        weights = np.full(n, 1.0 / n)

    consistent_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if weights[j] > 1e-9:
                consistent_matrix[i, j] = weights[i] / weights[j]
            else:
                consistent_matrix[i, j] = 1.0
    return consistent_matrix
