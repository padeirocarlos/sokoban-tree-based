"""
Workflow nodes for the deep research vehicles agent.

This module contains all the node functions that implement the core
logic of the agentic vehicles workflow.
"""
import os
import copy
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from graph.states import SokobanState
from agent.agent import SokobanAgentic
from sokoban.sokoban_tools import create_gif

logger = logging.getLogger("Sokoban-Agentic-Workflow (SAW)")

sokobanAgentic = SokobanAgentic()

record_path = "result/running_record.csv"
temp_result_folder = Path.cwd() / "result"
if not os.path.exists(temp_result_folder):
    os.makedirs(temp_result_folder)
    
def add_graph_attributes(span, node_id: str, parent_id: str = None):
    span.set_attribute("graph.node.id", node_id)
    if parent_id:
        span.set_attribute("graph.node.parent_id", parent_id)
    return span

def format_conversation( messages: List[Any], current_answer:str = "") -> str: 
    conversation = "Conversation history:\n\n"
    for message in messages:
        if isinstance(message, dict):
            conversation += f"{message.get("role")}: {message.get("content")}\n"
    
    if current_answer:
        conversation += f"assistant: {current_answer}\n"
    return conversation

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
        
        state_to_map = state["map"].convert_current_state_to_map()
        description_map = state["map"].description_map_state()
        user_query = f"{state_to_map} \n {description_map}"
        
        if int(state['current_iteration']) >= 2:
            user_query = f"{state_to_map} \n {description_map} \n This previous proposed solution steps, which did not solved the game, can you improve it : \n {state["previous_solution"][-1]}"
            
        # Call the model and record the execution time, and model name
        # gpt-oss:20b llama3:latest mistral:latest ollama3 qwen3 ayansh03/agribot
        # result = await sokobanAgentic.sokoban_react_agent(user_query=user_query, model_name=model_name) 
        result = await sokobanAgentic.sokoban_reflection_agent(user_query=user_query, model_name=model_name) 
        
        logger.info(f""" 🔀 1.0 Inputs maps: \n {user_query} \n """)
        logger.info(f""" 🔀 1.1 Plan_chunk: \n {result["answers"]} \n """)
        
        state["previous_solution"].append(result["answers"])
        plan_result = sokobanAgentic.post_processing_moves(result["answers"])
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        state['records']['model'] = "mistral"

        if plan_result:
            state['moves'] = plan_result
        
        logger.info(f""" 🔀 1.2 Plan_chunk: {plan_result} \n """)
        state['records']['time'] = execution_time

        refined_time = (time.perf_counter() - start_time) * 1000
        logger.info(f""" 🔀 ✅ Move_NODE: Executed success full in {float(refined_time):.2f} ms """)
            
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
        
        sokobanRules = copy.deepcopy(state['map'])
        total_moves = ""
        
        if state['moves'] == "":
            state['status'] = "empty"
            return state
            
        for move in state['moves']:
            if sokobanRules.makeMove(move):
                total_moves += move
                cur_map_state = sokobanRules.serialize_map()
                
                if sokobanRules.isLevelFinished():
                    state['status'] = "success"
                    state['visited_map_state'].append(cur_map_state)
                    state['solution'] = total_moves
                    state['final_response'] = total_moves
                    logger.info(f""" 🔀 Executor_NODE: Level Finished move {move} / {total_moves} """)
                else:
                    if cur_map_state in state['visited_map_state']:
                        idx = state['visited_map_state'].index(cur_map_state)
                        total_moves = total_moves[ :idx+1]
                        state['visited_map_state'] = state['visited_map_state'][: idx+1]
                        logger.info(f""" 🔀 Executor_NODE: Enter into repeated MODE for {state['current_iteration']}/{state['max_iterations']} 🧠 """)
                        continue 
                    else:
                        state['visited_map_state'].append(cur_map_state)
                        logger.info(f""" 🔀 Executor_NODE: NOT Visited_map_state {move} / {total_moves} """)
                        
                    state['status'] = "unsolved"
            else:
                state["moves"] = ""
                state["status"] = "invalid"
                logger.info(f""" 🔀 Executor_NODE: ❌ WRONG move {move} / {total_moves} """)
        
        logger.info(f""" Executor_NODE: GameStateObj: {state['map'].gameStateObj['boxes']} |  LevelObj: {state['map'].levelObj} | total_moves: {total_moves} """)
        refined_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"""  ✅ Executor_NODE: Executor NODE Executed | State: {state["status"]} Duration: {float(refined_time):.2f} ms 🧠 """)
        
        return {**state, }
    except Exception as e:
        logger.error(f"❌ Executor NODE failed: {e}")
        return {**state, }

async def result_node(state: SokobanState) -> SokobanState:
    """
    Record the running information to save to dataframe.
    Record the running steps to show dynamic map change.
    """
    if str(state['status']).lower() == ("success").lower():
        logger.info("Result_NODE: 🔬 🚀Congratulation🚀 You solved the puzzle: ", state['moves'])        
    elif str(state['status']).lower() == ("end").lower():
        logger.warning(f" 🔬 ⚠️ Result_NODE: The AI fails to solve it. Try it later 🏄🏽")
        
    # record the running meta data for model comparison
    file_name = state['map'].data_file
    state['records']['data_file']  = file_name
    state['records']['moves'] = state["moves"]
    state['records']['result'] = state['status']
    state['records']['iteration'] = state['current_iteration']
    
    logger.info("state['records]:", state['records'])
    
    new_df = pd.DataFrame([state['records']])
    
    if os.path.exists(record_path):
        existing_df = pd.read_csv(record_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(record_path, index=False)
    else:
        new_df.to_csv(record_path, index=False)
        
    # record each step in the dynamic map.    
    with open(temp_result_folder / f"result_map_{file_name}", 'w') as f:
        f.write(state['map'].serialize_map())
        f.write("\n")
        f.write("\n")
        
        for map in state['visited_map_state']:
            f.write(map)
            f.write("\n")
            f.write("\n")
            
    logger.info("Save the dynamic steps in result.txt.")
    
    # save the dynamic map as gif
    create_gif(state['visited_map_state'], 
                filename=temp_result_folder / f"result_map_{file_name[:-4]}.gif", 
                success=(state['status']=="success"),
                logger=logger)
            
    return state

async def router_node(state: SokobanState) -> SokobanState:
    pass
   