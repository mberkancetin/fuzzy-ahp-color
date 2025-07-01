import pytest
import json
import numpy as np

from multiAHPy.model import Node, Alternative, Hierarchy
from multiAHPy.pipeline import Workflow
from multiAHPy.types import TFN, Crisp

# ==============================================================================
# 1. FIXTURES: Reusable JSON data
# ==============================================================================

@pytest.fixture
def basic_hierarchy_json_str() -> str:
    """A JSON string representing a simple hierarchy and alternatives."""
    data = {
        "hierarchy": {
            "id": "Goal",
            "description": "Select the best option.",
            "children": [
                {"id": "Criterion A"},
                {"id": "Criterion B"}
            ]
        },
        "alternatives": [
            {"name": "Option 1", "description": "The first choice."},
            {"name": "Option 2"}
        ]
    }
    return json.dumps(data)


@pytest.fixture
def full_workflow_json_str() -> str:
    """A complete JSON defining an entire 'ranking' workflow."""
    data = {
        "workflow_config": {
            "workflow_type": "ranking",
            "group_strategy": "aggregate_judgments",
            "recipe": {
                "number_type": "TFN",
                "aggregation_method": "geometric",
                "weight_derivation_method": "geometric_mean"
            }
        },
        "hierarchy": {
            "id": "Goal",
            "children": [{"id": "Price"}, {"id": "Quality"}]
        },
        "alternatives": [{"name": "Car A"}, {"name": "Car B"}],
        "expert_judgments": [
            {
                "expert_id": "expert_1",
                "weight": 0.6,
                "matrices": {
                    "Goal": [[0.5]],        # Price vs Quality
                    "Price": [[5]],       # Car A vs Car B on Price
                    "Quality": [[0.25]]   # Car A vs Car B on Quality
                }
            },
            {
                "expert_id": "expert_2",
                "weight": 0.4,
                "matrices": {
                    "Goal": [[2]],
                    "Price": [[4]],
                    "Quality": [[0.5]]
                }
            }
        ]
    }
    return json.dumps(data)

# ==============================================================================
# 2. UNIT TESTS
# ==============================================================================

# --- Low-Level Serialization Tests ---

def test_node_to_dict():
    """Tests that a Node object can be correctly serialized to a dictionary."""
    root = Node("A")
    root.add_child(Node("B", "Child B"))
    root.add_child(Node("C"))

    expected = {
        "id": "A",
        "description": None,
        "children": [
            {"id": "B", "description": "Child B", "children": []},
            {"id": "C", "description": None, "children": []}
        ]
    }
    assert root.to_dict() == expected

def test_node_from_dict():
    """Tests that a Node object can be correctly created from a dictionary."""
    data = {
        "id": "A",
        "description": "Root Node",
        "children": [{"id": "B"}]
    }
    node = Node.from_dict(data)

    assert isinstance(node, Node)
    assert node.id == "A"
    assert node.description == "Root Node"
    assert len(node.children) == 1
    assert node.children[0].id == "B"
    assert node.children[0].parent == node

def test_hierarchy_from_json(basic_hierarchy_json_str):
    """Tests that a Hierarchy model can be constructed from a JSON string."""
    model = Hierarchy.from_json(basic_hierarchy_json_str, number_type=Crisp)

    assert isinstance(model, Hierarchy)
    assert model.number_type == Crisp
    assert model.root.id == "Goal"
    assert len(model.root.children) == 2
    assert len(model.alternatives) == 2
    assert model.alternatives[0].name == "Option 1"
    assert model.alternatives[0].description == "The first choice."

def test_hierarchy_from_malformed_json():
    """Tests that from_json raises an error for invalid JSON."""
    malformed_json = '{"hierarchy": {"id": "Goal"}}' # Missing "alternatives" key
    with pytest.raises(ValueError, match="JSON string must contain 'hierarchy' and 'alternatives' keys."):
        Hierarchy.from_json(malformed_json, number_type=Crisp)

# --- High-Level Workflow Tests ---

def test_workflow_from_json_and_run(full_workflow_json_str):
    """
    Tests the end-to-end functionality of creating and running a workflow from a single JSON.
    This is the most important integration test.
    """
    # This single call should parse the JSON, create the pipeline, load all data,
    # run the full calculation, and return the fitted object.
    pipeline = Workflow.from_json(full_workflow_json_str)

    # --- Assertions on the final state of the pipeline ---
    assert isinstance(pipeline, Workflow)

    # Check that results were generated
    assert pipeline.rankings is not None, "Rankings should be calculated"
    assert pipeline.criteria_weights is not None, "Criteria weights should be calculated"
    assert pipeline.consistency_report is not None, "Consistency report should be generated"

    # Check the configuration was parsed correctly
    assert pipeline.workflow_type == "ranking"
    assert pipeline.recipe['number_type'] == TFN

    # Check the results
    assert len(pipeline.rankings) == 2
    # Based on the data (Price favors A, Quality favors B, but Price is more important),
    # Car A should have a higher score.
    rank_dict = dict(pipeline.rankings)
    assert rank_dict["Car A"] > rank_dict["Car B"]

    # Check that the consistency report for the aggregated model was created
    assert "aggregated_model" in pipeline.consistency_report
    assert "Goal" in pipeline.consistency_report["aggregated_model"]

def test_workflow_from_json_with_invalid_type():
    """Tests that an unsupported number_type in the JSON recipe raises an error."""
    invalid_json_data = {
        "workflow_config": {"recipe": {"number_type": "ImaginaryFuzzyNumber"}},
        "hierarchy": {"id": "Goal", "children": []},
        "alternatives": []
    }
    invalid_json_str = json.dumps(invalid_json_data)

    with pytest.raises(ValueError, match="Unsupported number_type in JSON recipe: 'ImaginaryFuzzyNumber'"):
        Workflow.from_json(invalid_json_str)
