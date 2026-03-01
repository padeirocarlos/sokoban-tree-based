"""
Routing logic for the Sokoban agentic workflow graph.
"""
import logging
from typing import Literal, Dict, Any

logger = logging.getLogger("Sokoban-Agentic-Workflow (SAW)")


def route_after_executor_node(state: Dict[str, Any]) -> Literal["moves", "result"]:
    """Decide whether to retry (back to moves) or finalize (to result)."""
    status = state['status']
    iteration = state['current_iteration']
    max_iter = state['max_iterations']

    if status in ("invalid", "unsolved", "empty"):
        if iteration >= max_iter:
            logger.info(f"No solution found | Status: {status} | Iterations: {iteration}/{max_iter}")
            return "result"
        logger.info(f"Retrying | Status: {status} | Iterations: {iteration}/{max_iter}")
        return "moves"

    if status == "success":
        logger.info(f"Puzzle solved | Iterations: {iteration}/{max_iter}")
    else:
        logger.warning(f"Unexpected status '{status}', routing to result")

    return "result"
