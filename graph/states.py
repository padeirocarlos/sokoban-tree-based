"""
Workflow nodes for the deep research Sokoban agent.

This module contains all the node functions that implement the core
logic of the agentic Sokoban workflow.
"""
from typing import Optional, TypedDict, List
from sokoban.sokoban_tools import SokobanRules

class SokobanState(TypedDict):
    """
    State for car agentic workflow and quality evaluation.
    Tracks query processing, research iterations, and performance metrics.
    """
    moves: str                          # All executed primitive moves (L/R/U/D)
    status: str                         # "unsolved", "continue", "invalid", "success", etc.
    records: dict                       # running metadata recording
    solution: str                       # solution steps (exclude those repetitive cyclic steps)
    model_name: str                     # the LLM model name on deepinfra
    map: SokobanRules                   # Current state of Sokoban game
    max_iterations: int                 # max allowed iteration times
    current_iteration: int              # current iteration index
    visited_map_state: List[str]        # visited maps (serialized)
    previous_solution: List[str]        # previous solution
    final_response: Optional[str]       # final response

def initiate_state(initial_sokoban: SokobanRules, model_name: str) -> SokobanState:

    return {
        "map": initial_sokoban,                    
        "status": "continue",                         
        "moves": "",                       
        "solution": "",
        "max_iterations": 1,
        "visited_map_state": [],
        "current_iteration": 0,
        "previous_solution": [],
        "records": {},
        "model_name": model_name #  gpt-oss:20b llama3:latest mistral:latest ollama3 qwen3 ayansh03/agribot
    }