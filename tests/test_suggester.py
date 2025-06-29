# New user workflow in run_custom_test.py

import multiAHPy as ahp
from multiAHPy.model import Node
from multiAHPy.types import TFN
from multiAHPy.matrix_builder import create_matrix_from_list

# --- 1. Define the Problem Structure ---
# Hierarchy
goal_node = Node("Select Best Car")
crit_price = Node("Price")
crit_safety = Node("Safety")
goal_node.add_child(crit_price)
goal_node.add_child(crit_safety)
# Alternatives
alternatives = ["Car A", "Car B", "Car C"]

# --- 2. Get a Recipe for the Analysis ---
# We want a simple, robust analysis
recipe = ahp.suggester.get_model_recipe(
    fuzzy_number_preference="simple",
    aggregation_goal="average",
    weight_derivation_goal="stable_and_simple"
)
number_type = recipe["number_type"]

from multiAHPy.types import IFN
# number_type = IFN
recipe_manual = {
    'number_type': number_type,
    'aggregation_method': 'ifwa',
    'weight_derivation_method': 'geometric_mean',
    'consistency_method': 'value'
 }

# --- 3. Create and Configure the Pipeline ---
# Let's do a classic ranking workflow
pipeline = ahp.Workflow(
    root_node=goal_node,
    workflow_type="ranking",
    recipe=recipe,
    # workflow_type="scoring",
    # recipe=recipe_manual,
    alternatives=alternatives
)

# --- 4. Provide the Data ---
# Judgments from two experts
# Expert 1
e1_crit = create_matrix_from_list([3], number_type)
e1_price = create_matrix_from_list([5, 2, 2/5], number_type)
e1_safety = create_matrix_from_list([1/4, 1/2, 2], number_type)
# Expert 2
e2_crit = create_matrix_from_list([2], number_type)
e2_price = create_matrix_from_list([4, 3, 3/4], number_type)
e2_safety = create_matrix_from_list([1/3, 1/3, 1], number_type)

# Structure the data for the pipeline
expert_data = {
    "Select Best Car": [e1_crit, e2_crit], # Criteria matrix
    "Price": [e1_price, e2_price],         # Alternative matrix for Price
    "Safety": [e1_safety, e2_safety]       # Alternative matrix for Safety
}

# --- 5. Run the Pipeline ---
pipeline.run(expert_matrices=expert_data)

# --- 6. Access the Results ---
print("\nFinal Rankings from Pipeline:")
print(pipeline.rankings)

print("\nConsistency Report from Pipeline:")
print(pipeline.consistency_report)
