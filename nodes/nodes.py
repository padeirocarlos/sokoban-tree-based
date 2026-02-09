"""
Workflow nodes for the deep research vehicles agent.

This module contains all the node functions that implement the core
logic of the agentic vehicles workflow.
"""
import os
import time
import logging
from graph.states import SokobanState
from sokoban.sokoban_tools import SokobanRules
from sokoban.sokoban_tools import global_sokobanGame as sokobanGame
from agent.agent import SokobanAgentic, makePlayerMove, convert_current_state_to_map

logger = logging.getLogger("Sokoban-Agentic-Workflow (SAW)")
sokobanAgentic = SokobanAgentic()

async def move_node(state: SokobanState) -> SokobanState:
    """
    Generates moves (sequence of primitive moves)
    based on the current map.
    """
    try:
        logger.warning(f""" 🔀 ⚠️ Move_NODE: Start the trial: {state['current_iteration'] + 1} """)
        start_time = time.perf_counter()
        
        state['current_iteration'] = state['current_iteration'] + 1
        state['moves'] = ""
        state['status'] = "continue"
        state['visited_map_state'] = []
        model_name = state['model_name'] 
        sokoban_game = convert_current_state_to_map(sokobanGame=sokobanGame)
        
        if int(state['current_iteration']) >= 2:
            sokoban_game = f"""\n {sokoban_game} \n This previous proposed solution steps, 
                                which did not solved the game, can you improve it : \n {state["previous_solution"][-1]}"""
            
        # Call the model and record the execution time, and model name
            # gpt-oss:20b llama3:latest mistral:latest ollama3 qwen3 ayansh03/agribot
        
        # result = await sokobanAgentic.sokoban_react_agent(sokoban_game=sokoban_game, model_name=model_name) 
        result = await sokobanAgentic.sokoban_reflection_agent(sokoban_game=sokoban_game, model_name=model_name) 
        
        plan_result = "".join(result["answers"])
        state["previous_solution"].append(plan_result)
        
        if plan_result:
            state['moves'] = plan_result
        
        refined_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"""📝 🔍 Move_NODE: Executed success full! Proposed Solution: {plan_result} | Inference Time: {float(refined_time):.2f} ms ✅""")
            
        return {**state,}
            
    except Exception as e:
        logger.error(f"❌ Moving NODE failed: {e}")
        return {**state, }

async def executor_node(state: SokobanState) -> SokobanState: 
    """
    Executes the moves issued by plan.
    """
    try:
        start_time = time.perf_counter()
        logger.info(f""" 🔀 🧠  Executor_NODE: Starting Executor NODE """)
        
        total_moves = ""
        if state['moves'] == "":
            state['status'] = "empty"
            return state
        
        sokobanRules = SokobanRules(state['test_file'])
        
        for move in state['moves']:
            move_result = makePlayerMove(player_moving=move, sokobanGame=sokobanRules)
            cur_map_state = convert_current_state_to_map(sokobanGame=sokobanRules)
            total_moves += move
            
            if "LEVEL_COMPLETED" in move_result:
                state['status'] = "success"
                state['visited_map_state'].append(cur_map_state)
                state['solution'] = total_moves
                state['final_response'] = total_moves
                logger.info(f""" 🔀 Executor_NODE: Level Finished move {move} / {total_moves} ✅""")
                break
            
            if "VALID_MOVE" in move_result:
                if cur_map_state in state['visited_map_state']:
                    idx = state['visited_map_state'].index(cur_map_state)
                    total_moves = total_moves[ :idx+1]
                    state['visited_map_state'] = state['visited_map_state'][: idx+1]
                    continue 
                else:
                    state['visited_map_state'].append(cur_map_state)
                state['status'] = "unsolved"
                state['final_response'] = total_moves
            
            else:
                state["moves"] = ""
                state['final_response'] = total_moves
                state["status"] = "invalid"
                
        refined_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"""🔀 🧠 Executor_NODE: Executor NODE Executed | State: {state["status"]} | Total Move: {len(total_moves)} | Solution: {total_moves} |  Duration: {float(refined_time):.2f} ms ✅""")
        
        return {**state, }
    except Exception as e:
        logger.error(f"❌ Executor NODE failed: {e}")
        return {**state, }

async def result_node(state: SokobanState) -> SokobanState:
    """
    Record the running information to save to dataframe.
    Record the running steps to show dynamic map change.
    """
    try:
        visited_map_state = "\n ------ \n".join(state['visited_map_state'])
        
        if str(state['status']).lower() == ("success").lower():
            logger.info(f"Result_NODE: 🔬 🚀 Congratulation 🚀 You solved the puzzle! | Puzzle State: {visited_map_state}  \n ------ \n | Puzzle Move: {state['moves']} \n")        
        else:
            logger.warning(f" 🔬 ⚠️ Result_NODE: The AI fails to solve it. Try it later 🏄🏽! | Puzzle State: {visited_map_state}  \n ------ \n | Puzzle Move: {state['moves']} \n")
        return state
    except Exception as e:
        logger.error(f"❌ Result NODE failed: {e}")
        return {**state, }