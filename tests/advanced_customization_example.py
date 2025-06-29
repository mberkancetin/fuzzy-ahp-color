import numpy as np
import multiAHPy as ahp
from multiAHPy.types import TFN, Number
from multiAHPy.model import Hierarchy, Node
from multiAHPy.matrix_builder import create_matrix_from_list
from multiAHPy.aggregation import aggregate_matrices, register_aggregation_method
from multiAHPy.weight_derivation import register_weight_method
from multiAHPy.consistency import register_consistency_method
from typing import List, Type, Dict, Any

# ==============================================================================
# PART 1: A RESEARCHER DEFINES THEIR CUSTOM METHODS AND PARAMETERS
#
# This section simulates what a user would add in their own script to extend
# the library's functionality.
# ==============================================================================

print("--- Defining and Registering Custom Components ---")

# 1.1: Custom Defuzzification Method
# A researcher wants a method that gives more weight to the lower (pessimistic) bound.
def pessimistic_leaning_mean(tfn_instance: TFN) -> float:
    """A custom defuzzification giving 2x weight to the lower bound."""
    return (2 * tfn_instance.l + tfn_instance.m + tfn_instance.u) / 4.0

# Register this new method directly to the TFN class
TFN.register_defuzzify_method('pessimistic_mean', pessimistic_leaning_mean)
print("✅ Registered custom defuzzification method: 'pessimistic_mean'")


# 1.2: Custom Consistency Index
# A simple custom index that finds the max value in the defuzzified matrix,
# as a rough measure of the largest single judgment.
@register_consistency_method("max_element_index")
def calculate_max_element_index(matrix: np.ndarray, consistency_method: str = 'centroid') -> float:
    """Calculates the maximum off-diagonal element in the defuzzified matrix."""
    n = matrix.shape[0]
    if n <= 1:
        return 1.0
    crisp_matrix = np.array([[cell.defuzzify(method=consistency_method) for cell in row] for row in matrix])
    np.fill_diagonal(crisp_matrix, -np.inf) # Ignore diagonal
    return np.max(crisp_matrix)

print("✅ Registered custom consistency index: 'max_element_index'")


# 1.3: Custom Weight Derivation Method
# A researcher proposes a "Row Sum Normalization" method.
@register_weight_method('TFN', 'row_sum_norm')
def row_sum_normalization_method(matrix: np.ndarray, number_type: Type[TFN], **kwargs) -> List[TFN]:
    """A custom method that derives weights based on simple row sums."""
    n = matrix.shape[0]
    row_sums = [np.sum(matrix[i, :]) for i in range(n)]
    total_sum = sum(row_sums, number_type.neutral_element())

    if abs(total_sum.defuzzify()) < 1e-9:
        return [number_type.neutral_element()] * n

    weights = [rs / total_sum for rs in row_sums]
    return weights

print("✅ Registered custom weight derivation method: 'row_sum_norm'")


# 1.4: Custom Aggregation Method
# A method that aggregates matrices by first defuzzifying them, averaging the crisp
# values, and then converting the result back to a fuzzy number (TFN).
@register_aggregation_method("TFN", "average_of_crisp")
def aggregate_average_of_crisp(matrices: List[np.ndarray], n: int, number_type: Type, weights: List[float]) -> np.ndarray:
    """Aggregates by averaging the defuzzified values."""
    if not matrices:
        raise ValueError("Matrix list cannot be empty.")

    crisp_matrices = [
        np.array([[cell.defuzzify(method='centroid') for cell in row] for row in m])
        for m in matrices
    ]
    # Calculate the weighted average of the crisp matrices
    avg_crisp_matrix = np.average(crisp_matrices, axis=0, weights=weights)

    # Convert the averaged crisp matrix back to the specified fuzzy number type
    from multiAHPy.matrix_builder import create_comparison_matrix, complete_matrix_from_upper_triangle
    fuzzy_matrix = create_comparison_matrix(n, number_type)
    for i in range(n):
        for j in range(i + 1, n):
            fuzzy_matrix[i, j] = number_type.from_crisp(avg_crisp_matrix[i, j])

    return complete_matrix_from_upper_triangle(fuzzy_matrix)

print("✅ Registered custom aggregation method: 'average_of_crisp'")
print("-" * 50)


# ==============================================================================
# PART 2: SETUP AHP MODEL
# ==============================================================================

# 2.1: Define Hierarchy Structure
goal_node = Node("Select Best Car")
crit_price = Node("Price")
crit_safety = Node("Safety")
goal_node.add_child(crit_price)
goal_node.add_child(crit_safety)

model = Hierarchy(goal_node, number_type=TFN)

# 2.2: Add Alternatives
model.add_alternative("Car A")
model.add_alternative("Car B")
model.add_alternative("Car C")

# 2.3: Define some sample fuzzy judgments for two "experts"
# Judgments are for upper triangle: [C1/C2] for criteria; [A1/A2, A1/A3, A2/A3] for alternatives
# Expert 1 is decisive
expert1_crit_judgments = [3]
expert1_price_judgments = [5, 2, 1] # Car A is much cheaper than B, slightly cheaper than C
expert1_safety_judgments = [1/4, 1/2, 1] # Car A is much less safe than B and C

# Expert 2 is less certain
expert2_crit_judgments = [2]
expert2_price_judgments = [4, 3, 2]
expert2_safety_judgments = [1/3, 1/3, 1]

# Create fuzzy matrices from these judgments
# We'll use the 'wide' scale to represent higher uncertainty
crit_matrix_e1 = create_matrix_from_list(expert1_crit_judgments, TFN, scale='wide')
price_matrix_e1 = create_matrix_from_list(expert1_price_judgments, TFN, scale='wide')
safety_matrix_e1 = create_matrix_from_list(expert1_safety_judgments, TFN, scale='wide')

crit_matrix_e2 = create_matrix_from_list(expert2_crit_judgments, TFN, scale='wide')
price_matrix_e2 = create_matrix_from_list(expert2_price_judgments, TFN, scale='wide')
safety_matrix_e2 = create_matrix_from_list(expert2_safety_judgments, TFN, scale='wide')


# ==============================================================================
# PART 3: RUN THE MODEL WITH DEFAULT SETTINGS
# ==============================================================================

print("\n--- 3. RUNNING WITH DEFAULT LIBRARY SETTINGS ---\n")

# Aggregate matrices using the default 'geometric' method
agg_crit_matrix = aggregate_matrices([crit_matrix_e1, crit_matrix_e2], method='geometric')
agg_price_matrix = aggregate_matrices([price_matrix_e1, price_matrix_e2], method='geometric')
agg_safety_matrix = aggregate_matrices([safety_matrix_e1, safety_matrix_e2], method='geometric')

# Set matrices on the model
model.set_comparison_matrix("Select Best Car", agg_crit_matrix)
model.set_alternative_matrix("Price", agg_price_matrix)
model.set_alternative_matrix("Safety", agg_safety_matrix)

# Calculate weights and scores using default methods
model.calculate_weights(method="geometric_mean")
model.calculate_alternative_scores(derivation_method="geometric_mean")

print("Rankings using default 'centroid' defuzzification:")
print(model.get_rankings())

print("\nConsistency check with default settings:")
consistency_results_default = model.check_consistency()
print(consistency_results_default['Select Best Car'])
print("-" * 50)


# ==============================================================================
# PART 4: RUN THE MODEL WITH CUSTOM SETTINGS AND CONFIGURATIONS
# ==============================================================================

print("\n--- 4. RUNNING WITH CUSTOM SETTINGS & CONFIGURATION ---\n")

# 4.1: Modify the global configuration
print("## 4.1. Modifying global configuration ##")
# Let's use Alonso & Lamata's RI values and a new "super_wide" scale
alonso_lamata_ri = {
    1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
    8: 1.41, 9: 1.45, 'default': 1.50
}
ahp.configure_parameters.SAATY_RI_VALUES = alonso_lamata_ri
print(f"Changed RI value for n=3 to: {ahp.configure_parameters.SAATY_RI_VALUES[3]}")

# Add a new fuzzy scale
super_wide_scale = {1: (1,1,3), 2: (1,2,5), 3: (2,3,7), 4: (3,4,8), 5:(4,5,9), 6:(5,6,9), 7:(6,7,9), 8:(7,8,9), 9:(8,9,9)}
ahp.configure_parameters.FUZZY_TFN_SCALES['super_wide'] = super_wide_scale
print(f"Added new fuzzy scale 'super_wide': {ahp.configure_parameters.FUZZY_TFN_SCALES['super_wide']}")


# 4.2: Re-run the analysis using custom components
print("\n## 4.2. Re-running analysis with custom methods ##")

# Aggregate matrices using our new 'average_of_crisp' method
print("\nAggregating with custom 'average_of_crisp' method...")
custom_agg_crit = aggregate_matrices([crit_matrix_e1, crit_matrix_e2], method='average_of_crisp')
custom_agg_price = aggregate_matrices([price_matrix_e1, price_matrix_e2], method='average_of_crisp')
custom_agg_safety = aggregate_matrices([safety_matrix_e1, safety_matrix_e2], method='average_of_crisp')

# Set the new aggregated matrices on the model
model.set_comparison_matrix("Select Best Car", custom_agg_crit)
model.set_alternative_matrix("Price", custom_agg_price)
model.set_alternative_matrix("Safety", custom_agg_safety)

# Calculate weights and scores using our new 'row_sum_norm' method
print("\nCalculating weights with custom 'row_sum_norm' method...")
model.calculate_weights(method="row_sum_norm")
model.calculate_alternative_scores(derivation_method="row_sum_norm")

# Get rankings using our new 'pessimistic_mean' defuzzification method
print("\nRankings using custom 'pessimistic_mean' defuzzification:")
print(model.get_rankings(consistency_method='pessimistic_mean'))

# Check consistency. The output should now include our custom index and use the new RI values.
print("\nConsistency check with custom settings:")
consistency_results_custom = model.check_consistency()
print(consistency_results_custom['Price']) # Show one result in detail

print("-" * 50)
print("\n🎉 Customization test complete! Notice the different rankings and consistency scores.")

# Optional: Reset config to defaults if you continue working in the same session
ahp.configure_parameters.reset_to_defaults()
print(f"\nConfiguration has been reset to default. RI for n=3 is now: {ahp.configure_parameters.SAATY_RI_VALUES[3]}")
