import pytest
import pandas as pd
import multiAHPy as ahp
from multiAHPy.model import Node
from multiAHPy.types import TFN, IFN
from multiAHPy.matrix_builder import create_matrix_from_list

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

def load_expert_matrices_from_file(file_path: str, number_type) -> dict:
    """Loads survey data and converts it into a dictionary of matrices."""
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
            matrix = create_matrix_from_list(judgments, number_type, scale='wide')
            expert_matrices[node_id].append(matrix)

    return expert_matrices

@pytest.fixture(scope="session")
def root_node():
    """Provides the hierarchy structure, shared for the entire test session."""
    return build_hierarchy()

@pytest.fixture(scope="session")
def raw_tfn_matrices():
    """Provides the raw, inconsistent expert judgments as TFN matrices."""
    return load_expert_matrices_from_file("tests/data/survey_judgments.csv", TFN)

@pytest.fixture(scope="session")
def raw_ifn_matrices():
    """Provides the raw, inconsistent expert judgments as IFN matrices."""
    return load_expert_matrices_from_file("tests/data/survey_judgments.csv", IFN)

@pytest.fixture(scope="session")
def performance_scores():
    """Loads and normalizes performance scores."""
    df = pd.read_csv("tests/data/performance_scores.csv").set_index("CompanyName")
    return (df / 100.0).to_dict(orient='index')
