import pytest
import multiAHPy as ahp
from multiAHPy.sanitization import DataSanitizer
from multiAHPy.types import TFN, IFN

# Use the fixtures defined in conftest.py
pytestmark = pytest.mark.filterwarnings("ignore:Values in x were outside bounds during a minimize step")

def test_compare_sanitization_impact_on_final_scores(root_node, raw_tfn_matrices, performance_scores):
    """
    An end-to-end test that compares the final scores produced by using
    raw data vs. sanitized data.
    """
    alternatives = list(performance_scores.keys())

    # --- 1. Run with RAW data ---
    pipeline_raw = ahp.Workflow(
        root_node=root_node,
        workflow_type="scoring",
        group_strategy="aggregate_priorities",
        recipe=ahp.suggester.get_model_recipe(
            fuzzy_number_preference="simple",
            weight_derivation_goal="stable_and_simple"
            ),
        alternatives=alternatives
    )
    pipeline_raw.fit_weights(expert_matrices=raw_tfn_matrices)
    pipeline_raw.score(performance_scores=performance_scores)
    scores_raw = dict(pipeline_raw.rankings)

    print("\n--- Final Scores (RAW Data) ---")
    for company, score in scores_raw.items():
        print(f"  - {company}: {score:.4f}")

    # --- 2. Run with SANITIZED data ---
    sanitizer = DataSanitizer(strategy="adjust_bounded", target_cr=0.1, max_cycles=30)
    sanitized_matrices, _ = sanitizer.transform(raw_tfn_matrices, root_node, TFN)

    pipeline_sanitized = ahp.Workflow(
        root_node=root_node,
        workflow_type="scoring",
        group_strategy="aggregate_priorities",
        recipe=ahp.suggester.get_model_recipe(
            fuzzy_number_preference="simple",
            weight_derivation_goal="stable_and_simple"
            ),
        alternatives=alternatives
    )
    pipeline_sanitized.fit_weights(expert_matrices=sanitized_matrices)
    pipeline_sanitized.score(performance_scores=performance_scores)
    scores_sanitized = dict(pipeline_sanitized.rankings)

    print("\n--- Final Scores (SANITIZED Data) ---")
    for company, score in scores_sanitized.items():
        print(f"  - {company}: {score:.4f}")

    # --- 3. Assertions ---
    # The scores should be different
    assert scores_raw != scores_sanitized, "Sanitization should change the final scores."

    # The ranking order might even change
    ranking_raw = [company for company, score in pipeline_raw.rankings]
    ranking_sanitized = [company for company, score in pipeline_sanitized.rankings]
    print(f"\nRanking (Raw): {ranking_raw}")
    print(f"Ranking (Sanitized): {ranking_sanitized}")
    # We can't be sure they will be different, but it's a possibility
    # assert ranking_raw != ranking_sanitized
