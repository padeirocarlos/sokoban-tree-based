"""
Workflow nodes for the Sokoban agentic solver.

This module contains the node functions that implement the core
logic of the LangGraph workflow: generating moves, executing them,
and recording results.
"""
import time
import logging
from graph.states import SokobanState
from sokoban.sokoban_tools import SokobanRules
from agent.agent import SokobanAgentic, make_player_move, convert_current_state_to_map

logger = logging.getLogger("Sokoban-Agentic-Workflow (SAW)")
sokoban_agentic = SokobanAgentic()


async def move_node(state: SokobanState) -> SokobanState:
    """Generate a sequence of moves via LLM reflection on the current map."""
    logger.warning(f"Move_NODE: Start trial {state['current_iteration'] + 1}")
    start_time = time.perf_counter()

    state['current_iteration'] += 1
    state['moves'] = ""
    state['status'] = "continue"
    state['visited_map_state'] = []

    sokoban_rules = SokobanRules(state['test_file'])
    sokoban_game = convert_current_state_to_map(sokoban_rules)

    if state['current_iteration'] >= 2:
        sokoban_game = (
            f"\n {sokoban_game} \n This previous proposed solution steps,"
            f" which did not solve the game, can you improve it:\n {state['previous_solution'][-1]}"
        )

    result = await sokoban_agentic.sokoban_reflection_agent(
        sokoban_game=sokoban_game,
        model_name=state['model_name'],
        sokoban_rules=sokoban_rules,
    )

    plan_result = "".join(result["answers"])
    state["previous_solution"].append(plan_result)
    if plan_result:
        state['moves'] = plan_result

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"Move_NODE: Proposed solution: {plan_result} | Time: {elapsed_ms:.2f} ms")

    return state


async def executor_node(state: SokobanState) -> SokobanState:
    """Execute the planned moves against the game engine and validate results."""
    start_time = time.perf_counter()
    logger.info("Executor_NODE: Starting execution")

    if not state['moves']:
        state['status'] = "empty"
        return state

    total_moves = ""
    sokoban_rules = SokobanRules(state['test_file'])

    for move in state['moves']:
        move_result = make_player_move(player_moving=move, sokoban_game=sokoban_rules)
        cur_map_state = convert_current_state_to_map(sokoban_game=sokoban_rules)
        total_moves += move

        if "LEVEL_COMPLETED" in move_result:
            state['status'] = "success"
            state['visited_map_state'].append(cur_map_state)
            state['solution'] = total_moves
            state['final_response'] = total_moves
            logger.info(f"Executor_NODE: Level finished at move {move} / {total_moves}")
            break

        if "VALID_MOVE" in move_result:
            if cur_map_state in state['visited_map_state']:
                idx = state['visited_map_state'].index(cur_map_state)
                total_moves = total_moves[:idx + 1]
                state['visited_map_state'] = state['visited_map_state'][:idx + 1]
                continue
            state['visited_map_state'].append(cur_map_state)
            state['status'] = "unsolved"
            state['final_response'] = total_moves
        else:
            state["moves"] = ""
            state['final_response'] = total_moves
            state["status"] = "invalid"

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"Executor_NODE: Status: {state['status']} | Moves: {len(total_moves)} "
        f"| Solution: {total_moves} | Time: {elapsed_ms:.2f} ms"
    )

    return state


async def result_node(state: SokobanState) -> SokobanState:
    """Log the final result of the puzzle-solving attempt."""
    visited_map_state = "\n ------ \n".join(state['visited_map_state'])

    if state['status'] == "success":
        logger.info(f"Result_NODE: Puzzle solved! | Moves: {state['moves']}")
    else:
        logger.warning(f"Result_NODE: AI failed to solve | Moves: {state['moves']}")

    return state