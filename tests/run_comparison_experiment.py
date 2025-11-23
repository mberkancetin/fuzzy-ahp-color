import multiAHPy as ahp
from multiAHPy.model import Node
from multiAHPy.types import TFN
from multiAHPy.sanitization import DataSanitizer
from multiAHPy.matrix_builder import create_matrix_from_list
import pandas as pd
import numpy as np
import os # To handle file paths

# ==============================================================================
# 0. HELPER FUNCTIONS FOR DATA LOADING AND ANALYSIS
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

def load_expert_matrices(file_path: str) -> dict:
    """Loads survey data and converts it into a dictionary of TFN matrices."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}. Make sure it's in a 'data' subdirectory.")

    df = pd.read_csv(file_path)
    num_experts = len(df)

    matrix_map = {
        "C1": [f"S1{i}/S1{j}" for i in range(1, 5) for j in range(i + 1, 5)],
        "C2": [f"S2{i}/S2{j}" for i in range(1, 5) for j in range(i + 1, 5)],
        "C3": [f"S3{i}/S3{j}" for i in range(1, 5) for j in range(i + 1, 5)],
        "C4": [f"S4{i}/S4{j}" for i in range(1, 5) for j in range(i + 1, 5)],
        "COLOR Index": [f"C{i}/C{j}" for i in range(1, 5) for j in range(i + 1, 5)]
    }

    expert_matrices = {node_id: [] for node_id in matrix_map}

    for i in range(num_experts):
        for node_id, cols in matrix_map.items():
            judgments = df.loc[i, cols].tolist()
            # Use 'wide' scale to represent the uncertainty of non-experts
            matrix = create_matrix_from_list(judgments, TFN, scale='wide')
            expert_matrices[node_id].append(matrix)

    return expert_matrices

def load_performance_scores(file_path: str) -> dict:
    """Loads and normalizes performance scores."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}.")
    df = pd.read_csv(file_path).set_index("CompanyName")
    return (df / 100.0).to_dict(orient='index')

def calculate_distortion(raw_matrices: dict, sanitized_matrices: dict) -> float:
    """Calculates the Mean Squared Error (MSE) between two sets of expert matrices."""
    total_squared_error = 0
    num_elements = 0
    num_experts = len(list(raw_matrices.values())[0])

    for i in range(num_experts):
        for node_id in raw_matrices.keys():
            raw_m = raw_matrices[node_id][i]
            san_m = sanitized_matrices[node_id][i]

            # Defuzzify both to get crisp representations for comparison
            raw_crisp = np.array([[c.defuzzify() for c in row] for row in raw_m])
            san_crisp = np.array([[c.defuzzify() for c in row] for row in san_m])

            total_squared_error += np.sum((raw_crisp - san_crisp)**2)
            num_elements += raw_crisp.size

    return total_squared_error / num_elements if num_elements > 0 else 0


if __name__ == "__main__":
    # ==========================================================================
    # 1. SETUP: Load data and define the problem
    # ==========================================================================

    print("--- Setting up Comparative Experiment ---")

    # Define file paths (assuming a 'data' folder in the same directory)
    SURVEY_DATA_PATH = "tests/data/survey_judgments.csv"
    PERFORMANCE_DATA_PATH = "tests/data/performance_scores.csv"

    # Load all data
    HIERARCHY_ROOT = build_hierarchy()
    RAW_TFN_MATRICES = load_expert_matrices(SURVEY_DATA_PATH)
    PERFORMANCE_SCORES = load_performance_scores(PERFORMANCE_DATA_PATH)
    ALTERNATIVES = list(PERFORMANCE_SCORES.keys())

    # Store results for final comparison
    comparison_results = []

    # ==========================================================================
    # 2. RUN SCENARIOS
    # ==========================================================================

    # --- SCENARIO 1: Baseline (No Revision) ---
    print("\n--- SCENARIO 1: Processing Raw, Inconsistent Data ---")
    pipeline_raw = ahp.Workflow(
        root_node=HIERARCHY_ROOT,
        workflow_type="scoring",
        group_strategy="aggregate_priorities", # AIP is a robust aggregation method
        recipe=ahp.suggester.get_model_recipe(fuzzy_number_preference="simple"),
        alternatives=ALTERNATIVES
    )
    pipeline_raw.fit_weights(expert_matrices=RAW_TFN_MATRICES)
    pipeline_raw.score(performance_scores=PERFORMANCE_SCORES)

    # Store results
    inconsistent_count = sum(1 for r in pipeline_raw.consistency_report.values() for m in r.values() if not m['is_consistent'])
    comparison_results.append({
        "Scenario": "Baseline (Raw Data)",
        "Distortion (MSE)": 0.0, # No changes made
        "Inconsistent Matrices": f"{inconsistent_count}/{len(RAW_TFN_MATRICES) * 8}",
        "Final Ranking": [r[0] for r in pipeline_raw.rankings],
        "Scores": {r[0]: r[1] for r in pipeline_raw.rankings}
    })

    # --- SCENARIO 2: GMR (Rebuild Consistent) ---
    print("\n--- SCENARIO 2: Sanitizing with 'rebuild_consistent' (GMR) ---")
    sanitizer_gmr = DataSanitizer(strategy="rebuild_consistent", target_cr=0.1, scale="wide")
    sanitized_matrices_gmr, _ = sanitizer_gmr.transform(RAW_TFN_MATRICES, HIERARCHY_ROOT, TFN)

    pipeline_gmr = ahp.Workflow(
        root_node=HIERARCHY_ROOT,
        workflow_type="scoring",
        group_strategy="aggregate_priorities",
        recipe=ahp.suggester.get_model_recipe(fuzzy_number_preference="simple"),
        alternatives=ALTERNATIVES
    )
    pipeline_gmr.fit_weights(expert_matrices=sanitized_matrices_gmr)
    pipeline_gmr.score(performance_scores=PERFORMANCE_SCORES)

    # Store results
    inconsistent_count_gmr = sum(1 for r in pipeline_gmr.consistency_report.values() for m in r.values() if not m['is_consistent'])
    distortion_gmr = calculate_distortion(RAW_TFN_MATRICES, sanitized_matrices_gmr)
    comparison_results.append({
        "Scenario": "Reconstruction (GMR)",
        "Distortion (MSE)": distortion_gmr,
        "Inconsistent Matrices": f"{inconsistent_count_gmr}/{len(RAW_TFN_MATRICES) * 8}",
        "Final Ranking": [r[0] for r in pipeline_gmr.rankings],
        "Scores": {r[0]: r[1] for r in pipeline_gmr.rankings}
    })

    # --- SCENARIO 3: PITR (Adjust Persistent) ---
    print("\n--- SCENARIO 3: Sanitizing with 'adjust_persistent' (PITR) ---")
    sanitizer_pitr = DataSanitizer(strategy="adjust_persistent", target_cr=0.1, max_cycles=50, scale="wide")
    sanitized_matrices_pitr, _ = sanitizer_pitr.transform(RAW_TFN_MATRICES, HIERARCHY_ROOT, TFN)

    pipeline_pitr = ahp.Workflow(
        root_node=HIERARCHY_ROOT,
        workflow_type="scoring",
        group_strategy="aggregate_priorities",
        recipe=ahp.suggester.get_model_recipe(fuzzy_number_preference="simple"),
        alternatives=ALTERNATIVES
    )
    pipeline_pitr.fit_weights(expert_matrices=sanitized_matrices_pitr)
    pipeline_pitr.score(performance_scores=PERFORMANCE_SCORES)

    # Store results
    inconsistent_count_pitr = sum(1 for r in pipeline_pitr.consistency_report.values() for m in r.values() if not m['is_consistent'])
    distortion_pitr = calculate_distortion(RAW_TFN_MATRICES, sanitized_matrices_pitr)
    comparison_results.append({
        "Scenario": "Persistent Revision (PITR)",
        "Distortion (MSE)": distortion_pitr,
        "Inconsistent Matrices": f"{inconsistent_count_pitr}/{len(RAW_TFN_MATRICES) * 8}",
        "Final Ranking": [r[0] for r in pipeline_pitr.rankings],
        "Scores": {r[0]: r[1] for r in pipeline_pitr.rankings}
    })

    # ==========================================================================
    # 3. FINAL REPORT
    # ==========================================================================

    print("\n\n" + "="*80)
    print("           COMPARATIVE EXPERIMENT RESULTS SUMMARY")
    print("="*80)

    # Create and format the final comparison DataFrame
    results_df = pd.DataFrame(comparison_results).set_index("Scenario")
    scores_df = results_df['Scores'].apply(pd.Series)
    results_df = pd.concat([results_df.drop('Scores', axis=1), scores_df], axis=1)

    # Format for better readability
    pd.options.display.float_format = '{:,.4f}'.format

    print(results_df)
    print("="*80)
    print(sanitized_matrices_pitr)
