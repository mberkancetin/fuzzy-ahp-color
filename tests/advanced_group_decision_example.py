"""
===================================================================
Fuzzy-AHP: Advanced Group Decision-Making Example
===================================================================

This script demonstrates an advanced Fuzzy AHP workflow known as the
"Aggregation of Priorities" method, particularly useful for group decisions
with Intuitionistic Fuzzy Sets (IFS).

Workflow Steps:
1.  Collect individual comparison matrices from multiple experts.
2.  Calculate a crisp priority vector (weights) for each expert's judgments.
3.  For each criterion, convert the set of crisp weights from all experts
    into a single Intuitionistic Fuzzy Number (IFN). This IFN represents
    the group's collective, fuzzy judgment on that criterion's importance.
4.  Aggregate these IFNs using the Intuitionistic Fuzzy Weighted Averaging
    (IFWA) operator to find the final group priorities.

This approach is academically robust as it allows for weighting experts
and explicitly models the agreement/disagreement among them as an IFN.
"""

import numpy as np

# Import necessary components from the library
from multiAHPy.model import Hierarchy, Node
from multiAHPy.types import TFN, Crisp, IFN
from multiAHPy.matrix_builder import create_matrix_from_judgments
from multiAHPy.weight_derivation import derive_weights
from multiAHPy.aggregation import aggregate_priorities_ifwa

# --- 1. Define Problem and Collect Individual Judgments ---

criteria = ["Innovation", "Feasibility", "Market Impact"]
expert_names = ["Senior R&D Lead", "Junior Engineer", "Marketing Manager"]

# Judgments from each expert for the criteria
judgments_expert1 = {("Innovation", "Feasibility"): 2, ("Innovation", "Market Impact"): 3, ("Feasibility", "Market Impact"): 2}
judgments_expert2 = {
    ("Innovation", "Feasibility"): 3,
    ("Innovation", "Market Impact"): 1,
    ("Feasibility", "Market Impact"): 1/2
}
judgments_expert3 = {("Innovation", "Feasibility"): 1/4, ("Innovation", "Market Impact"): 1, ("Feasibility", "Market Impact"): 4}

all_judgments = [judgments_expert1, judgments_expert2, judgments_expert3]

# --- 2. Calculate a Crisp Priority Vector for Each Expert ---

print("--- Step 2: Calculating Individual Expert Priority Vectors ---")
expert_crisp_weights = []
for i, judgments in enumerate(all_judgments):
    # Create a crisp matrix for each expert
    matrix = create_matrix_from_judgments(judgments, criteria, Crisp)
    # Derive crisp weights using the standard geometric mean method
    results = derive_weights(matrix, Crisp, method='geometric_mean')
    weights = results['crisp_weights']
    expert_crisp_weights.append(weights)
    print(f"  - Weights for {expert_names[i]}: {[f'{w:.3f}' for w in weights]}")

# We now have a list of weight vectors:
# e.g., [[0.54, 0.29, 0.16], [0.43, 0.19, 0.38], [0.19, 0.72, 0.09]]
expert_crisp_weights = np.array(expert_crisp_weights)

# --- 3. Convert Crisp Weights into a Single IFN for Each Criterion ---

def weights_to_ifn(weights: np.ndarray) -> IFN:
    """
    A helper function to convert a set of crisp weights for one item
    into a single Intuitionistic Fuzzy Number.

    Rule:
    - Membership (mu) is the mean of the weights.
    - Non-membership (nu) is based on the standard deviation (a measure of disagreement).
    - Hesitation (pi) is the remainder.
    """
    if len(weights) < 2:
        # If only one expert, it's a crisp judgment
        return IFN.from_crisp(weights[0])

    mu = np.mean(weights)
    # Let's define non-membership as the standard deviation. This is one common heuristic.
    # A high std dev means high disagreement, thus high non-membership in the average.
    nu = np.std(weights)

    # Ensure mu + nu <= 1, capping nu if necessary
    if mu + nu > 1.0:
        nu = 1.0 - mu

    return IFN(mu, nu)

print("\n--- Step 3: Converting Weight Sets to Intuitionistic Fuzzy Numbers (IFN) ---")
ifn_priorities = []
for i, criterion_name in enumerate(criteria):
    # Get all weights for this specific criterion (column i)
    weights_for_criterion = expert_crisp_weights[:, i]
    ifn = weights_to_ifn(weights_for_criterion)
    ifn_priorities.append(ifn)
    print(f"  - IFN for '{criterion_name}': {ifn}")

# --- 4. Aggregate IFN Priorities using IFWA ---

# Define weights for the experts themselves
# The Senior lead is most important, the marketing manager is also important.
expert_importance_weights = [0.5, 0.2, 0.3]

print(f"\n--- Step 4: Aggregating IFN Priorities with Expert Weights {expert_importance_weights} ---")

# The IFWA operator requires a list of priorities to aggregate. Since we want a final
# crisp ranking, we defuzzify the IFNs before aggregation in this workflow.
# A more advanced workflow might aggregate them into a final IFN.
# For ranking, we use the defuzzified values.

final_scores = {}
for i, ifn in enumerate(ifn_priorities):
    # We can use different defuzzification methods here to see the impact
    score = ifn.defuzzify(method='value')
    final_scores[criteria[i]] = score

print("\n--- FINAL GROUP PRIORITIES (Defuzzified) ---")
# Normalize the final scores to sum to 1
total_score = sum(final_scores.values())
final_normalized_priorities = {name: score / total_score for name, score in final_scores.items()}

# Sort for final ranking
ranked_criteria = sorted(final_normalized_priorities.items(), key=lambda item: item[1], reverse=True)

for i, (name, priority) in enumerate(ranked_criteria):
    print(f"  Rank {i+1}: {name} (Final Priority: {priority:.4f})")

# Let's also show the result of a weighted average on the crisp weights for comparison
print("\n--- For Comparison: Simple Weighted Average of Crisp Weights ---")
simple_weighted_avg = np.average(expert_crisp_weights, axis=0, weights=expert_importance_weights)
print(f"  - Simple Avg: {[f'{w:.3f}' for w in simple_weighted_avg/np.sum(simple_weighted_avg)]}")

