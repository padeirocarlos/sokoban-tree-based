"""
State definitions for the Sokoban agentic workflow.
"""
from typing import Optional, TypedDict, List


class SokobanState(TypedDict):
    """Tracks game state, iterations, and results across the LangGraph workflow."""
    moves: str
    status: str
    solution: str
    test_file: str
    model_name: str
    max_iterations: int
    current_iteration: int
    visited_map_state: List[str]
    previous_solution: List[str]
    final_response: Optional[str]


def initiate_state(model_name: str, test_file: str) -> SokobanState:
    """Create a fresh workflow state for a new puzzle-solving attempt."""
    return {
        "moves": "",
        "solution": "",
        "status": "continue",
        "max_iterations": 1,
        "visited_map_state": [],
        "current_iteration": 0,
        "previous_solution": [],
        "test_file": test_file,
        "model_name": model_name,
    }