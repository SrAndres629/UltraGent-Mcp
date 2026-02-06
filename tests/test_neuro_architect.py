import pytest
from unittest.mock import Mock, patch
from neuro_architect import NeuroArchitect, NeuronState, ImpactPrediction, SynapseType
import networkx as nx
import logging
import json
from dataclasses import asdict

@pytest.fixture
def neuro_architect():
    return NeuroArchitect()

@pytest.fixture
def neuron_state():
    return NeuronState()

@pytest.fixture
def impact_prediction():
    return ImpactPrediction("target_node", ["direct_impact"], ["ripple_effect"], 50.0, ["breaking_paths"])

def test_neuro_architect_init(neuro_architect):
    assert isinstance(neuro_architect._lock, type(Lock()))
    assert isinstance(neuro_architect._graph, nx.DiGraph)
    assert isinstance(neuro_architect._states, dict)

def test_neuron_state_to_dict(neuron_state):
    neuron_state.last_active = "2022-01-01T00:00:00"
    neuron_state.activation_level = 0.5
    neuron_state.error_rate = 0.1
    neuron_state.active_variables = {"var1": "value1"}
    neuron_state.logs = ["log1", "log2", "log3", "log4", "log5", "log6"]
    neuron_state.memories = [{"memory1": "value1"}]
    neuron_state.fix_attempts = 1
    expected_dict = {
        "last_active": "2022-01-01T00:00:00",
        "activation_level": 0.5,
        "error_rate": 0.1,
        "active_variables": {"var1": "value1"},
        "logs": ["log2", "log3", "log4", "log5", "log6"],
        "fix_attempts": 1
    }
    assert asdict(neuron_state) == expected_dict

def test_impact_prediction_to_dict(impact_prediction):
    expected_dict = {
        "target_node": "target_node",
        "direct_impact": ["direct_impact"],
        "ripple_effect": ["ripple_effect"],
        "risk_score": 50.0,
        "breaking_paths": ["breaking_paths"],
        "affected_nodes": ["direct_impact", "ripple_effect"]
    }
    assert asdict(impact_prediction) == expected_dict

@patch("neuro_architect.get_vision")
def test_neuro_architect_initialize_cortex(mock_get_vision, neuro_architect):
    mock_get_vision.return_value.scan_project.return_value = (["node1", "node2"], [("node1", "node2")])
    mock_get_vision.return_value.build_graph.return_value = nx.DiGraph()
    neuro_architect._initialize_cortex()
    assert len(neuro_architect._graph.nodes()) == 2
    assert len(neuro_architect._states) == 2

def test_neuro_architect_enrich_graph_with_data_flow(neuro_architect):
    # This test is currently incomplete as it requires the implementation of the _enrich_graph_with_data_flow method
    pass

@patch("neuro_architect.logging")
def test_neuro_architect_initialize_cortex_error(mock_logging, neuro_architect):
    mock_get_vision = Mock()
    mock_get_vision.scan_project.side_effect = Exception("Test error")
    with patch("neuro_architect.get_vision", return_value=mock_get_vision):
        neuro_architect._initialize_cortex()
        mock_logging.error.assert_called_once()

def test_synapse_type():
    assert SynapseType.IMPORT == "import"
    assert SynapseType.CALL == "call"
    assert SynapseType.DATA_FLOW == "data_flow"
    assert SynapseType.INHERITANCE == "inheritance"

def test_neuro_architect_init_project_root(neuro_architect):
    assert neuro_architect._project_root is None
    neuro_architect = NeuroArchitect(project_root="test_project")
    assert neuro_architect._project_root == "test_project"

def test_neuro_architect_logger(neuro_architect):
    assert isinstance(neuro_architect._vision, object)
    assert isinstance(neuro_architect._lock, type(Lock()))
    assert isinstance(neuro_architect._graph, nx.DiGraph)
    assert isinstance(neuro_architect._states, dict)