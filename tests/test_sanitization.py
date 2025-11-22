import pytest
import multiAHPy as ahp
from multiAHPy.sanitization import DataSanitizer, SANITIZATION_STRATEGIES
from multiAHPy.types import TFN, Crisp

# Use the fixtures defined in conftest.py
pytestmark = pytest.mark.filterwarnings("ignore:Values in x were outside bounds during a minimize step")
TARGET_CR = 0.1

def test_sanitizer_registry():
    """Tests that all sanitization strategies are correctly registered."""
    assert "adjust_persistent" in SANITIZATION_STRATEGIES
    assert "adjust_persistent" in SANITIZATION_STRATEGIES
    assert "rebuild_consistent" in SANITIZATION_STRATEGIES

def test_sanitizer_init_fails_with_bad_strategy():
    """Tests that the DataSanitizer raises an error for an unknown strategy."""
    with pytest.raises(ValueError, match="Unknown sanitization strategy 'bad_strategy'"):
        DataSanitizer(strategy="bad_strategy")

@pytest.mark.parametrize("strategy", ["adjust_persistent", "adjust_persistent"])
def test_iterative_sanitization_strategies(strategy, root_node, raw_tfn_matrices):
    """
    Tests the iterative sanitization strategies ('adjust_persistent', 'adjust_persistent')
    to ensure they produce a consistent set of matrices.
    """
    print(f"\n--- Testing Sanitizer with strategy: '{strategy}' ---")

    # 1. Configure the sanitizer
    sanitizer = DataSanitizer(
        strategy=strategy,
        target_cr=0.1,
        max_cycles=30, # Give it enough cycles to work
        bound=9.0
    )

    # 2. Run the transformation
    sanitized_matrices, change_log = sanitizer.transform(
        raw_expert_matrices=raw_tfn_matrices,
        root_node=root_node,
        target_number_type=TFN
    )

    # 3. Verify the results
    assert sanitized_matrices is not None, "Sanitization should return a matrix dictionary."
    assert "expert_1" in change_log, "Change log should be populated for revised experts."

    # 4. CRITICAL: Verify that the output matrices are now consistent.
    # We do this by running a final consistency check on the sanitized data.
    final_pipeline = ahp.Workflow(
        root_node=root_node,
        workflow_type='scoring',
        recipe=ahp.suggester.get_model_recipe(
            fuzzy_number_preference="simple",
            weight_derivation_goal="stable_and_simple"
            ),
        alternatives=['temp'],
        group_strategy='aggregate_priorities'
    )
    final_pipeline.fit_weights(expert_matrices=sanitized_matrices)

    for expert_id, report in final_pipeline.consistency_report.items():
        for node_id, metrics in report.items():
            cr_val = metrics.get('saaty_cr')
            assert isinstance(cr_val, float), f"CR for {expert_id}/{node_id} is not a number: {cr_val}"
            assert cr_val <= TARGET_CR, \
                f"Matrix for {expert_id}/{node_id} failed to meet CR target. " \
                f"Final CR: {cr_val:.4f}"

    print(f"✅ Strategy '{strategy}' successfully produced matrices with CR <= {TARGET_CR}.")

def test_rebuild_consistent_strategy(root_node, raw_tfn_matrices):
    """
    Tests the one-shot 'rebuild_consistent' strategy.
    """
    print("\n--- Testing Sanitizer with strategy: 'rebuild_consistent' ---")

    sanitizer = DataSanitizer(
        strategy="rebuild_consistent",
        target_cr=0.1,
        completion_method="llsm" # Use the fast method
    )

    sanitized_matrices, change_log = sanitizer.transform(
        raw_expert_matrices=raw_tfn_matrices,
        root_node=root_node,
        target_number_type=TFN
    )

    assert "info" in change_log, "Change log should contain info about the rebuild."

    # Verify that the output matrices are now consistent
    final_pipeline = ahp.Workflow(
        root_node=root_node,
        workflow_type='scoring',
        recipe=ahp.suggester.get_model_recipe(
            fuzzy_number_preference="simple",
            weight_derivation_goal="stable_and_simple"
            ),
        alternatives=['temp'],
        group_strategy='aggregate_priorities'
    )
    final_pipeline.fit_weights(expert_matrices=sanitized_matrices)

    for expert_id, report in final_pipeline.consistency_report.items():
        for node_id, metrics in report.items():
            cr_val = metrics.get('saaty_cr')
            assert isinstance(cr_val, float), f"CR for {expert_id}/{node_id} is not a number: {cr_val}"
            assert cr_val <= TARGET_CR, \
                f"Matrix for {expert_id}/{node_id} failed to meet CR target. " \
                f"Final CR: {cr_val:.4f}"

    print(f"✅ Strategy 'rebuild_consistent' successfully produced matrices with CR <= {TARGET_CR}.")
