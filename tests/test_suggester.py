# tests/test_suggester.py

import pytest
import multiAHPy as ahp
from multiAHPy.model import Node
from multiAHPy.pipeline import Workflow
from multiAHPy.types import TFN, TrFN, IFN

# ==============================================================================
# 1. FIXTURES
# ==============================================================================

@pytest.fixture
def basic_hierarchy():
    """Provides a simple, reusable hierarchy structure for tests."""
    goal = Node("Goal")
    goal.add_child(Node("Criterion 1"))
    goal.add_child(Node("Criterion 2"))
    return goal

@pytest.fixture
def basic_alternatives():
    """Provides a simple list of alternative names."""
    return ["Option A", "Option B"]

# ==============================================================================
# 2. TESTS FOR suggester
# ==============================================================================

def test_suggester_get_available_options():
    """
    Tests that the suggester can report its available options correctly.
    """
    options = ahp.suggester.get_available_options()
    assert "fuzzy_number_preference" in options
    assert "simple" in options["fuzzy_number_preference"]
    assert "hesitation" in options["fuzzy_number_preference"]
    assert "aggregation_goal" in options

def test_suggester_simple_average_recipe():
    """
    Tests the default recipe for a simple average model.
    """
    recipe = ahp.suggester.get_model_recipe(
        fuzzy_number_preference="simple",
        aggregation_goal="average",
        weight_derivation_goal="stable_and_simple"
    )

    assert recipe['number_type'] == TFN
    assert recipe['aggregation_method'] == 'geometric'
    assert recipe['weight_derivation_method'] == 'geometric_mean'
    assert recipe['consistency_method'] == 'graded_mean'

def test_suggester_robust_recipe():
    """
    Tests the recipe generation for a robust model that handles outliers.
    """
    recipe = ahp.suggester.get_model_recipe(
        fuzzy_number_preference="interval_certainty",
        aggregation_goal="robust_to_outliers",
        weight_derivation_goal="true_fuzzy"
    )

    assert recipe['number_type'] == TrFN
    assert recipe['aggregation_method'] == 'median'
    assert recipe['weight_derivation_method'] == 'lambda_max'

def test_suggester_ifn_recipe():
    """
    Tests the recipe generation specifically for an IFN-based model.
    """
    recipe = ahp.suggester.get_model_recipe(
        fuzzy_number_preference="hesitation",
        aggregation_goal="consensus"
    )

    assert recipe['number_type'] == IFN
    assert recipe['aggregation_method'] == 'consensus'
    assert recipe['consistency_method'] == 'centroid'

def test_suggester_invalid_input():
    """
    Tests that the suggester raises a ValueError for invalid preference strings.
    """
    with pytest.raises(ValueError, match="Invalid fuzzy_number_preference"):
        ahp.suggester.get_model_recipe(fuzzy_number_preference="invalid_choice")

    with pytest.raises(ValueError, match="Invalid aggregation_goal"):
        ahp.suggester.get_model_recipe(aggregation_goal="invalid_choice")


# ==============================================================================
# 3. TESTS FOR PIPELINE INITIALIZATION USING SUGGESTER RECIPES
# ==============================================================================

def test_pipeline_initialization_with_tfn_recipe(basic_hierarchy, basic_alternatives):
    """
    Tests creating a pipeline with a TFN-based recipe.
    This directly validates the fix for the previous collection error.
    """
    recipe = ahp.suggester.get_model_recipe(fuzzy_number_preference="simple")

    try:
        pipeline = Workflow(
            root_node=basic_hierarchy,
            workflow_type="scoring",
            group_strategy="aggregate_judgments",
            recipe=recipe,
            alternatives=basic_alternatives
        )
    except Exception as e:
        pytest.fail(f"Pipeline initialization failed with TFN recipe: {e}")

    assert isinstance(pipeline, Workflow)
    assert pipeline.model.number_type == TFN
    assert pipeline.recipe['number_type'] == TFN
    assert len(pipeline.model.alternatives) == 2
    assert pipeline.model.alternatives[0].name == "Option A"


def test_pipeline_initialization_with_ifn_recipe(basic_hierarchy, basic_alternatives):
    """
    Tests creating a pipeline with an IFN-based recipe.
    """
    recipe = ahp.suggester.get_model_recipe(fuzzy_number_preference="hesitation")

    try:
        pipeline = Workflow(
            root_node=basic_hierarchy,
            workflow_type="ranking",
            group_strategy="aggregate_priorities",
            recipe=recipe,
            alternatives=basic_alternatives
        )
    except Exception as e:
        pytest.fail(f"Pipeline initialization failed with IFN recipe: {e}")

    assert isinstance(pipeline, Workflow)
    assert pipeline.model.number_type == IFN
    assert pipeline.recipe['number_type'] == IFN
    assert pipeline.group_strategy == "aggregate_priorities"
