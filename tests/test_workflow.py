import pytest
from backend.graph.workflow import build_workflow


def test_workflow_is_compiled():
    graph = build_workflow()
    assert graph is not None
