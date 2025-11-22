import multiAHPy as ahp
from multiAHPy.model import Node
from multiAHPy.types import TFN, IFN
import pandas as pd

# ==============================================================================
# 0. HELPER FUNCTIONS (to load and prepare data)
# ==============================================================================

def build_hierarchy():
    """Builds the specific hierarchy for the corporate responsibility model."""
    goal = Node("COLOR Index")
    criteria = {}
    for i in range(1, 5):
        c_id = f"C{i}"
        criteria[c_id] = Node(c_id)
        goal.add_child(criteria[c_id])
        for j in range(1, 5):
            s_id = f"S{i}{j}"
            criteria[c_id].add_child(Node(s_id))
    return goal

def load_expert_matrices(file_path: str, number_type) -> dict:
    """Loads survey data and converts it into a dictionary of matrices for the pipeline."""
    df = pd.read_csv(file_path)
    num_experts = len(df)

    matrix_map = {
        "C1": ["S11/S12", "S11/S13", "S11/S14", "S12/S13", "S12/S14", "S13/S14"],
        "C2": ["S21/S22", "S21/S23", "S21/S24", "S22/S23", "S22/S24", "S23/S24"],
        "C3": ["S31/S32", "S31/S33", "S31/S34", "S32/S33", "S32/S34", "S33/S34"],
        "C4": ["S41/S42", "S41/S43", "S41/S44", "S42/S43", "S42/S44", "S43/S44"],
        "COLOR Index": ["C1/C2", "C1/C3", "C1/C4", "C2/C3", "C2/C4", "C3/C4"]
    }

    expert_matrices = {node_id: [] for node_id in matrix_map}

    for i in range(num_experts):
        for node_id, cols in matrix_map.items():
            judgments = df.loc[i, cols].tolist()
            # Use a 'wide' scale for TFN to introduce uncertainty.
            # IFN has its own built-in scale.
            matrix = ahp.matrix_builder.create_matrix_from_list(judgments, number_type, scale='wide')
            expert_matrices[node_id].append(matrix)

    return expert_matrices

def load_performance_scores(file_path: str) -> dict:
    """Loads and normalizes performance scores."""
    df = pd.read_csv(file_path).set_index("CompanyName")
    normalized_df = df / 100.0
    return normalized_df.to_dict(orient='index')

def print_final_report(pipeline: ahp.Workflow, title: str):
    """A helper function to print a formatted summary of the results."""
    print("\n--- Final Consistency Report (RAW) ---")
    import json
    print(json.dumps(pipeline.consistency_report, indent=2))
    print("\n" + "="*80)
    print(f"RESULTS FOR: {title}")
    print("="*80)

    # 4. Final Consistency Report
    print("\n--- Final Consistency Report ---")
    if pipeline.group_strategy == 'aggregate_priorities':
        print("(Showing consistency for each of the 8 experts)")
        # Summarize the AIP report
        for expert_id, report in pipeline.consistency_report.items():
            inconsistent_nodes = [node for node, metrics in report.items() if not metrics['is_consistent']]
            if inconsistent_nodes:
                print(f"  - {expert_id}: {len(inconsistent_nodes)} inconsistent matrices (e.g., {inconsistent_nodes[0]})")
            else:
                print(f"  - {expert_id}: All matrices are consistent.")
    else: # aggregate_judgments
        print("(Showing consistency of the single aggregated model)")
        report = pipeline.consistency_report.get('aggregated_model', {})
        for node_id, metrics in report.items():
            status = "CONSISTENT" if metrics['is_consistent'] else f"INCONSISTENT (CR={metrics.get('saaty_cr', 'N/A'):.4f})"
            print(f"  - {node_id}: {status}")

    # 5. Final Scores
    print("\n--- Final Company Scores (COLOR Index) ---")
    if pipeline.rankings:
        df = pd.DataFrame(pipeline.rankings, columns=['Company', 'Score']).set_index('Company')
        print(df.sort_values(by='Score', ascending=False))
    else:
        print("No rankings were calculated.")
    print("="*80 + "\n")


# ==============================================================================
# 1. SETUP: Load data and define the problem
# ==============================================================================

# Define the common structure and data for all scenarios
HIERARCHY_ROOT = build_hierarchy()
PERFORMANCE_SCORES = load_performance_scores("tests/data/performance_scores.csv")
ALTERNATIVES = list(PERFORMANCE_SCORES.keys())

# Load expert judgments for both IFN and TFN types
EXPERT_MATRICES_IFN = load_expert_matrices("tests/data/survey_judgments.csv", IFN)
EXPERT_MATRICES_TFN = load_expert_matrices("tests/data/survey_judgments.csv", TFN)

# ==============================================================================
# 2. SCENARIO ANALYSIS
# ==============================================================================

# --- SCENARIO 1: IFN (Main Choice) with Robust Aggregation & Revision ---
# This is the most academically sound approach for this dataset.
# GOAL: Compare the two revision strategies.
# ------------------------------------------------------------------------------

# 1a. IFN with "adjust_bounded" revision
"""title_1a = 'IFN | Aggregate Priorities | "adjust_bounded" Revision'
pipeline_1a = ahp.Workflow(
    root_node=HIERARCHY_ROOT,
    workflow_type="scoring",
    group_strategy="aggregate_priorities",
    recipe=ahp.suggester.get_model_recipe(fuzzy_number_preference="hesitation"),
    alternatives=ALTERNATIVES
)
pipeline_1a.fit_weights(expert_matrices=EXPERT_MATRICES_IFN)
pipeline_1a.make_consistent(strategy="adjust_bounded", max_cycles=50)
pipeline_1a.score(performance_scores=PERFORMANCE_SCORES)
print_final_report(pipeline_1a, title_1a)"""

# 1b. TFN with "remove_and_complete" revision
title_1b = 'TFN | Aggregate Priorities | "remove_and_complete" Revision'
pipeline_1b = ahp.Workflow(
    root_node=HIERARCHY_ROOT,
    workflow_type="scoring",
    group_strategy="aggregate_priorities",
    recipe=ahp.suggester.get_model_recipe(fuzzy_number_preference="simple"),
    alternatives=ALTERNATIVES
)
pipeline_1b.fit_weights(expert_matrices=EXPERT_MATRICES_TFN)
pipeline_1b.make_consistent(strategy="remove_and_complete", completion_method="llsm", max_cycles=50)
pipeline_1b.score(performance_scores=PERFORMANCE_SCORES)
print_final_report(pipeline_1b, title_1b)


# --- SCENARIO 2: Compare Number Types (IFN vs TFN) ---
# GOAL: See how much the choice of fuzzy number affects the outcome,
# using the same robust process for both.
# ------------------------------------------------------------------------------
title_2 = 'TFN | Aggregate Priorities | "remove_and_complete" Revision'
pipeline_2 = ahp.Workflow(
    root_node=HIERARCHY_ROOT,
    workflow_type="scoring",
    group_strategy="aggregate_priorities",
    recipe=ahp.suggester.get_model_recipe(fuzzy_number_preference="simple"), # Using "simple" for TFN
    alternatives=ALTERNATIVES
)
pipeline_2.fit_weights(expert_matrices=EXPERT_MATRICES_TFN)
# pipeline_2.make_consistent(strategy="remove_and_complete", max_cycles=50)
pipeline_2.score(performance_scores=PERFORMANCE_SCORES)
print_final_report(pipeline_2, title_2)


# --- SCENARIO 3: Compare Aggregation Methods (AIP vs AIJ) ---
# GOAL: See the difference between aggregating priorities vs. aggregating judgments.
# We will use IFN and no consistency revision to see the raw effect.
# ------------------------------------------------------------------------------
"""title_3 = "IFN | Aggregate Judgments (AIJ) | No Revision"
pipeline_3 = ahp.Workflow(
    root_node=HIERARCHY_ROOT,
    workflow_type="scoring",
    group_strategy="aggregate_judgments", # <-- The key change
    recipe={'number_type': IFN, 'aggregation_method': 'ifwa', 'weight_derivation_method': 'geometric_mean'},
    alternatives=ALTERNATIVES
)
pipeline_3.fit_weights(expert_matrices=EXPERT_MATRICES_IFN)
pipeline_3.score(performance_scores=PERFORMANCE_SCORES)
print_final_report(pipeline_3, title_3)

print("\n\nANALYSIS COMPLETE. Compare the final scores across the different reports.")
"""
