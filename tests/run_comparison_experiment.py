import multiAHPy as ahp
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from multiAHPy.model import Node
from multiAHPy.types import Crisp, TFN
from multiAHPy.sanitization import DataSanitizer
from multiAHPy.matrix_builder import create_matrix_from_list
from multiAHPy.weight_derivation import derive_weights
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
    """Loads survey data and converts it into a dictionary of Crisp matrices."""
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
            # Use 'linear' scale to represent the uncertainty of non-experts
            matrix = create_matrix_from_list(judgments, Crisp, scale='linear')
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

def calculate_mae(raw_crisp, san_crisp):
    return np.mean(np.abs(raw_crisp - san_crisp))

def calculate_cosine_similarity(raw_matrix, sanitized_matrix):
    raw_weights = derive_weights(raw_matrix, Crisp, method="geometric_mean")['crisp_weights']
    sanitized_weights = derive_weights(sanitized_matrix, Crisp, method="geometric_mean")['crisp_weights']

    # scipy.spatial.distance.cosine calculates the distance (1 - similarity)
    return 1 - cosine(raw_weights, sanitized_weights)

def calculate_spearman_correlation(raw_matrix, sanitized_matrix):
    # 1. Calculate weights for both matrices
    raw_weights = derive_weights(raw_matrix, Crisp, method="geometric_mean")['crisp_weights']
    sanitized_weights = derive_weights(sanitized_matrix, Crisp, method="geometric_mean")['crisp_weights']

    # 2. Calculate the correlation of the ranks of the weights
    correlation, _ = spearmanr(raw_weights, sanitized_weights)
    return correlation

def calculate_cosine_similarity(raw_matrix, sanitized_matrix):
    raw_weights = derive_weights(raw_matrix, Crisp, method="geometric_mean")['crisp_weights']
    sanitized_weights = derive_weights(sanitized_matrix, Crisp, method="geometric_mean")['crisp_weights']

    # scipy.spatial.distance.cosine calculates the distance (1 - similarity)
    return 1 - cosine(raw_weights, sanitized_weights)

def analyze_distortion(raw_matrices: dict, sanitized_matrices: dict) -> dict:
    """
    Calculates multiple distortion and similarity metrics between two sets of matrices.
    """
    total_squared_error = 0
    total_absolute_error = 0
    num_elements = 0

    correlations = []
    similarities = []

    num_experts = len(list(raw_matrices.values())[0])

    for i in range(num_experts):
        for node_id in raw_matrices.keys():
            raw_m_obj = raw_matrices[node_id][i]
            san_m_obj = sanitized_matrices[node_id][i]

            raw_crisp = np.array([[c.defuzzify() for c in row] for row in raw_m_obj])
            san_crisp = np.array([[c.defuzzify() for c in row] for row in san_m_obj])

            # Element-wise errors
            total_squared_error += np.sum((raw_crisp - san_crisp)**2)
            total_absolute_error += np.sum(np.abs(raw_crisp - san_crisp))
            num_elements += raw_crisp.size

            # Weight-vector based similarity
            raw_weights = derive_weights(raw_m_obj, Crisp, method="geometric_mean")['crisp_weights']
            san_weights = derive_weights(san_m_obj, Crisp, method="geometric_mean")['crisp_weights']

            corr, _ = spearmanr(raw_weights, san_weights)
            sim = 1 - cosine(raw_weights, san_weights)

            if not np.isnan(corr): correlations.append(corr)
            if not np.isnan(sim): similarities.append(sim)

    return {
        "Distortion (MSE)": total_squared_error / num_elements if num_elements > 0 else 0,
        "Distortion (MAE)": total_absolute_error / num_elements if num_elements > 0 else 0,
        "Avg. Rank Correlation (Spearman)": np.mean(correlations) if correlations else 0,
        "Avg. Weight Similarity (Cosine)": np.mean(similarities) if similarities else 0,
    }

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
    RAW_Crisp_MATRICES = load_expert_matrices(SURVEY_DATA_PATH)
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
        recipe={
            "number_type": Crisp,
            'weight_derivation_method': 'geometric_mean',
            'consistency_method': 'centroid'
            },
        alternatives=ALTERNATIVES
    )
    pipeline_raw.fit_weights(expert_matrices=RAW_Crisp_MATRICES)
    pipeline_raw.score(performance_scores=PERFORMANCE_SCORES)

    # Store results
    inconsistent_count = sum(1 for r in pipeline_raw.consistency_report.values() for m in r.values() if not m['is_consistent'])
    distortion_metrics_raw = analyze_distortion(RAW_Crisp_MATRICES, RAW_Crisp_MATRICES)
    comparison_results.append({
        "Scenario": "Baseline (Raw Data)",
        **distortion_metrics_raw,
        "Inconsistent Matrices": f"{inconsistent_count}/{len(RAW_Crisp_MATRICES) * 32}",
        "Final Ranking": [r[0] for r in pipeline_raw.rankings],
        "Scores": {r[0]: r[1] for r in pipeline_raw.rankings}
    })

    # --- SCENARIO 2: GMR (Rebuild Consistent) ---
    print("\n--- SCENARIO 2: Sanitizing with 'rebuild_consistent' (GMR) ---")
    sanitizer_gmr = DataSanitizer(strategy="rebuild_consistent", target_cr=0.1, max_cycles=500, scale="linear")
    sanitized_matrices_gmr, _ = sanitizer_gmr.transform(RAW_Crisp_MATRICES, HIERARCHY_ROOT, Crisp)

    pipeline_gmr = ahp.Workflow(
        root_node=HIERARCHY_ROOT,
        workflow_type="scoring",
        group_strategy="aggregate_priorities",
        recipe={
            "number_type": Crisp,
            'weight_derivation_method': 'geometric_mean',
            'consistency_method': 'centroid'
            },
        alternatives=ALTERNATIVES
    )
    pipeline_gmr.fit_weights(expert_matrices=sanitized_matrices_gmr)
    pipeline_gmr.score(performance_scores=PERFORMANCE_SCORES)

    # Store results
    inconsistent_count_gmr = sum(1 for r in pipeline_gmr.consistency_report.values() for m in r.values() if not m['is_consistent'])
    distortion_gmr = calculate_distortion(RAW_Crisp_MATRICES, sanitized_matrices_gmr)
    distortion_metrics_gmr = analyze_distortion(RAW_Crisp_MATRICES, sanitized_matrices_gmr)
    comparison_results.append({
        "Scenario": "Reconstruction (GMR)",
        **distortion_metrics_gmr,
        "Inconsistent Matrices": f"{inconsistent_count_gmr}/{len(RAW_Crisp_MATRICES) * 32}",
        "Final Ranking": [r[0] for r in pipeline_gmr.rankings],
        "Scores": {r[0]: r[1] for r in pipeline_gmr.rankings}
    })

    # --- SCENARIO 3: Simple Iterative (Simple Iterative) ---
    print("\n--- SCENARIO 3: Sanitizing with 'simple_iterative' ---")
    sanitizer_si = DataSanitizer(strategy="adjust_bounded", target_cr=0.1, max_cycles=500, scale="linear")
    sanitized_matrices_si, _ = sanitizer_si.transform(RAW_Crisp_MATRICES, HIERARCHY_ROOT, Crisp)

    pipeline_si = ahp.Workflow(
        root_node=HIERARCHY_ROOT,
        workflow_type="scoring",
        group_strategy="aggregate_priorities",
        recipe={
            "number_type": Crisp,
            'weight_derivation_method': 'geometric_mean',
            'consistency_method': 'centroid'
            },
        alternatives=ALTERNATIVES
    )
    pipeline_si.fit_weights(expert_matrices=sanitized_matrices_si)
    pipeline_si.score(performance_scores=PERFORMANCE_SCORES)

    # Store results
    inconsistent_count_si = sum(1 for r in pipeline_si.consistency_report.values() for m in r.values() if not m['is_consistent'])
    distortion_si = calculate_distortion(RAW_Crisp_MATRICES, sanitized_matrices_si)
    distortion_metrics_si = analyze_distortion(RAW_Crisp_MATRICES, sanitized_matrices_si)
    comparison_results.append({
        "Scenario": "Simple Iterative",
        **distortion_metrics_si,
        "Inconsistent Matrices": f"{inconsistent_count_si}/{len(RAW_Crisp_MATRICES) * 32}",
        "Final Ranking": [r[0] for r in pipeline_si.rankings],
        "Scores": {r[0]: r[1] for r in pipeline_si.rankings}
    })


   # --- SCENARIO 4: PITR (Adjust Persistent) ---
    print("\n--- SCENARIO 4: Sanitizing with 'adjust_persistent' (PITR) ---")
    sanitizer_pitr = DataSanitizer(strategy="adjust_persistent", target_cr=0.1, max_cycles=500, scale="linear")
    sanitized_matrices_pitr, _ = sanitizer_pitr.transform(RAW_Crisp_MATRICES, HIERARCHY_ROOT, Crisp)

    pipeline_pitr = ahp.Workflow(
        root_node=HIERARCHY_ROOT,
        workflow_type="scoring",
        group_strategy="aggregate_priorities",
        recipe={
            "number_type": Crisp,
            'weight_derivation_method': 'geometric_mean',
            'consistency_method': 'centroid'
            },
        alternatives=ALTERNATIVES
    )
    pipeline_pitr.fit_weights(expert_matrices=sanitized_matrices_pitr)
    pipeline_pitr.score(performance_scores=PERFORMANCE_SCORES)


    # Store results
    inconsistent_count_pitr = sum(1 for r in pipeline_pitr.consistency_report.values() for m in r.values() if not m['is_consistent'])
    distortion_metrics_pitr = analyze_distortion(RAW_Crisp_MATRICES, sanitized_matrices_pitr)
    comparison_results.append({
        "Scenario": "Persistent Revision (PITR)",
        **distortion_metrics_pitr,
        "Inconsistent Matrices": f"{inconsistent_count_pitr}/{len(RAW_Crisp_MATRICES) * 32}",
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

    print(results_df["Distortion (MSE)"])
    print(results_df["Distortion (MAE)"])
    print(results_df["Avg. Rank Correlation (Spearman)"])
    print(results_df["Avg. Weight Similarity (Cosine)"])

    print(results_df["Inconsistent Matrices"])
    print(results_df["Company A"])
    print(results_df["Company B"])
    print(results_df["Company C"])

    print("="*80)
