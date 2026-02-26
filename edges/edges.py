"""
Routing and edge logic for the deep research agent workflow.

This module contains the conditional routing functions that determine
the flow of execution through the agentic workflow graph.
"""
import logging
from typing import Literal, Dict, Any
logger = logging.getLogger("Sokoban-Agentic-Workflow (SAW)")

def route_after_executor_node(state: Dict[str, Any]) -> Literal[ "moves", "result"]:
    """
    This routing agent determines the next move step after predict a move,
    and system state.
    
    :param state: Description
    :type state: Dict[str, Any]
    :rtype: Literal["END", "moves", "result"]
    """
    if state['status'] == "invalid" or state['status'] == "unsolved" or state['status'] == "empty":
        if state['current_iteration'] >= state['max_iterations']:
            logger.info(f" 📝 Could not find the solution | Status: {state['status']} | Iterations: {state['current_iteration']}/{state['max_iterations']} 🔍")
            return "result"
        else:
            logger.info(f" ❌ Could not find solution at trial | Status: {state['status']} | Iterations: {state['current_iteration']}/{state['max_iterations']}")
            return "moves"
        
    elif state['status'] == "success":
        logger.info(f" 📝 Sokoban successful executed | Status: {state['status']} | Iterations: {state['current_iteration']}/{state['max_iterations']} ✅")
        return "result"

    else:
        logger.warning(f"Unexpected status '{state['status']}', routing to result")
        return "result"
