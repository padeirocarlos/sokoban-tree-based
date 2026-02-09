import torch
import logging
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from typing import Dict,Any,List,Union
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from .output_entity import GeneralResult
from langchain_core.outputs import LLMResult

from langchain_core.runnables import RunnableSequence
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain.agents.structured_output import ToolStrategy
from sokoban.sokoban_tools import global_sokobanGame as sokobanGame
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.tools import render_text_description,tool, BaseTool, StructuredTool
from .instructions import sokoban_assist_template, sokoban_system_template, sokoban_reflection_template

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

load_dotenv(override=True)
logger = logging.getLogger("Sokoban-Agentic-Workflow")

def convert_current_state_to_map() -> str:
    map_width = sokobanGame.levelObj['width']
    map_height = sokobanGame.levelObj['height']
    map = sokobanGame.levelObj['mapObj']
    starting_map = sokobanGame.levelObj['mapObj']
    
    for i in range(map_height):
        for j in range(map_width):
            if starting_map[i][j] == "#":
                map[i][j] = "#"
            # updated player and stars position
            elif starting_map[i][j] == ".":
                if (i, j) == sokobanGame.gameStateObj['player']:
                    map[i][j] = "+"
                elif (i, j) in sokobanGame.gameStateObj['boxes']:
                    map[i][j] = "*"
                else:
                    map[i][j] = "."
            elif (i, j) == sokobanGame.gameStateObj['player']:
                    map[i][j] = "@"
            elif (i, j) in sokobanGame.gameStateObj['boxes']:
                    map[i][j] = "$"
            else:
            # other places are just floor
                map[i][j] = " "
        
    game_map = "\n".join("".join(row) for row in map)
    player = f"\n Player (@) position: {sokobanGame.gameStateObj['player']}"
    box = f"\n Box ($) positions: {sokobanGame.gameStateObj['boxes']}"
    target = f"\n Target (.) positions: {sokobanGame.targets}"
    
    return f"{game_map} \n {player} {box} {target}"

@tool
def _makePlayerMove(player_moving: str) -> str:
    """
    Attempt to move the player in a specified direction in the Sokoban game.
    
    This function validates and executes a player move, handling both simple 
    movements and box-pushing mechanics. The move succeeds only if the target 
    position is not blocked by a wall, or if pushing a box, the space behind 
    the box is also free.
    
    Parameters:
        player_moving (str): Direction to move the player. Accepts one formats:
            - Up: 'U' (case-insensitive)
            - Down: 'D' (case-insensitive)
            - Left: 'L' (case-insensitive)
            - Right: 'R' (case-insensitive)
    
    Returns:
        str: A message describing the result of the move attempt:
            - Success: "The player's new position is (x, y)"
            - Failure: Reason why the move failed (wall blocking or box cannot be pushed)
    
    Behavior:
        1. Validates the direction input
        2. Checks if the target position contains a wall (move fails)
        3. If target contains a box, checks if the box can be pushed:
           - If space behind box is free: pushes box and moves player
           - If space behind box is blocked: move fails
        4. Updates game state if move is successful
        5. Check if the current Sokoban level has been completed.
        6. Check if a specific grid position contains a wall in the Sokoban game.
        7. Check if a grid position is blocked and cannot be moved into.
    
    Examples:
        makePlayerMove('U')
        makePlayerMove('R')
        makePlayerMove('D')
    Note:
        This function modifies the game state (player position and box positions)
        when a move is successful.
    """
    return makePlayerMove(player_moving)

def makePlayerMove(player_moving: str) -> str:
    """
    Attempt to move the player in a specified direction in the Sokoban game.
    
    This function validates and executes a player move, handling both simple 
    movements and box-pushing mechanics. The move succeeds only if the target 
    position is not blocked by a wall, or if pushing a box, the space behind 
    the box is also free.
    
    Parameters:
        player_moving (str): Direction to move the player. Accepts one formats:
            - Up: 'U' (case-insensitive)
            - Down: 'D' (case-insensitive)
            - Left: 'L' (case-insensitive)
            - Right: 'R' (case-insensitive)
    
    Returns:
        str: A message describing the result of the move attempt:
            - Success: "The player's new position is (x, y)"
            - Failure: Reason why the move failed (wall blocking or box cannot be pushed)
    
    Behavior:
        1. Validates the direction input
        2. Checks if the target position contains a wall (move fails)
        3. If target contains a box, checks if the box can be pushed:
           - If space behind box is free: pushes box and moves player
           - If space behind box is blocked: move fails
        4. Updates game state if move is successful
        5. Check if the current Sokoban level has been completed.
        6. Check if a specific grid position contains a wall in the Sokoban game.
        7. Check if a grid position is blocked and cannot be moved into.
    
    Examples:
        makePlayerMove('U')
        makePlayerMove('R')
        makePlayerMove('D')
    Note:
        This function modifies the game state (player position and box positions)
        when a move is successful.
    """
    
    def isWall(x, y):
        """Returns True if the (x, y) position on
        the map is a wall, otherwise return False."""
        if x < 0 or x >= len(sokobanGame.mapObj) or y < 0 or y >= len(sokobanGame.mapObj[x]):
            return False # x and y aren't actually on the map.
        elif sokobanGame.mapObj[x][y] in ('#', 'x'):
            return True # wall is blocking
        return False

    def isBlocked( x, y):
        """Returns True if the (x, y) position on the map is
        blocked by a wall or star, otherwise return False."""

        if isWall(x, y):
            return True

        elif x < 0 or x >= len(sokobanGame.mapObj) or y < 0 or y >= len(sokobanGame.mapObj[x]):
            return True # x and y aren't actually on the map.

        elif (x, y) in sokobanGame.gameStateObj['boxes']:
            return True # a box is blocking
        return False
    
    def isLevelFinished():
        """Returns True if all the goals have stars in them."""
        for goal in sokobanGame.levelObj['goals']:
            if goal not in sokobanGame.gameStateObj['boxes']:
                return goal
        return True
    
    # Get current player position and boxes
    playerx, playery = sokobanGame.gameStateObj['player']
    boxes = sokobanGame.gameStateObj['boxes']
    
    is_LevelCompleted_Check = isLevelFinished()
    
    if isinstance(is_LevelCompleted_Check, bool):
        return "LEVEL_COMPLETED"
    
    # Parse movement direction and calculate offset
    direction = str(player_moving).upper()
    
    if "<U>" in direction or "(U)" in direction or "UP" in direction or "U" in direction or "'U'" in direction or "**U**" in direction: 
        xOffset = 0
        yOffset = -1
    elif "<D>" in direction or "(D)" in direction or "DOWN" in direction or "D" in direction or "'D'" in direction or "**D**" in direction:
        xOffset = 0
        yOffset = 1
    elif "<L>" in direction or "(L)" in direction or "LEFT" in direction or "L" in direction or "'L'" in direction or  "**L**" in direction: 
        xOffset = -1
        yOffset = 0
    elif "<R>" in direction or "(R)" in direction or "RIGHT" in direction or "R" in direction or "'R'" in direction or  "**R**" in direction:
        xOffset = 1
        yOffset = 0
    else:
        return f"Invalid direction: '{player_moving}'. Use U/D/L/R or UP/DOWN/LEFT/RIGHT"
    # Calculate target position
    target_x = playerx + xOffset
    target_y = playery + yOffset

    # Check if target position is a wall
    if isWall(target_x, target_y):
        return f"Cannot move, because the player's new position ({target_x}, {target_y}) is a wall, try a different move"
    
    # Check if target position contains a box
    if (target_x, target_y) in boxes:
        # Calculate position behind the box
        box_push_x = target_x + xOffset
        box_push_y = target_y + yOffset
        
        # Check if box can be pushed
        if isBlocked(box_push_x, box_push_y):
            # Push the box
            box_index = boxes.index((target_x, target_y))
            boxes[box_index] = (box_push_x, box_push_y)
            sokobanGame.gameStateObj['boxes'] = boxes
            # Move player to target position
            sokobanGame.gameStateObj['player'] = (target_x, target_y)
            return f"It is VALID_MOVE, the Player's new position is ({sokobanGame.gameStateObj['player'][0]}, {sokobanGame.gameStateObj['player'][1]}) \n the box's new position {','.join([str(({box[0]}, {box[1]})) for box in boxes])} \n" 
        else:
            return f"Cannot move, because the box's new position ({target_x}, {target_y}) is blocked, try a different move"
    
    # Move player to target position
    sokobanGame.gameStateObj['player'] = (target_x, target_y)
    return f"It is VALID_MOVE, the player's new position is ({sokobanGame.gameStateObj['player'][0]}, {sokobanGame.gameStateObj['player'][1]}) \n"

class AgentCallbackHandler(BaseCallbackHandler):
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        human = prompts[0].split("Human:")[1] # I need to get only Human prompt which is RecAct agent
        logger.info(f"🔀 Agent_Calling: Prompt to LLM \n {human} \n")

    # def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
    #     """Run when LLM ends running."""
    #     logger.info(f" 🔀 Agent_Calling: Final Result \n {response.generations[0][0].text} ms ✅ \n ")
        
class SokobanAgentic:
    
    def __init__(self, model_name: str="llama3"):
        self.model_name = model_name
    
    
    async def sokoban_reflection_agent(self, user_query:str , model_name:str) -> dict: 
        """
        Docstring for sokoban_agent
        
        :param user_query: Description
        :param model_name: Description
        :param prompt_type: Description
        :return: Description
        """

        LEVEL_COMPLETED = False
        sokoban_game_solution = []
        content = "Your task is to solve the sokoban game"
        tools = [_makePlayerMove]
        # tools = [isWall, isBlocked, isSokobanGameFinished, makePlayerMove]
        
        template_reflection_assist = sokoban_reflection_template(sokoban_game_state=user_query)
        
        generation_llm = ChatOllama(name = "Sokoban-Assistant-Agent", 
                                    model = model_name, 
                                    callbacks = [AgentCallbackHandler()], 
                                    max_iterations = 1,
                                    temperature=0.5)
        
        if "llama3" not in model_name: # ollama3 does not allow banding tool
            llm_bind_tools = generation_llm.bind_tools(tools)
        else:
            llm_bind_tools = generation_llm
            
        generation_chain = llm_bind_tools 
        messages = [HumanMessage(content=template_reflection_assist)]
        
        while not LEVEL_COMPLETED:
            
            result = await generation_chain.ainvoke(messages)
            
            current_state_map = convert_current_state_to_map() # current game state before the moving
            sokoban_game_result = self.reflection_processing_moves(result.content, sokoban_game_solution)
            sokoban_game_state = convert_current_state_to_map() + "\n" +sokoban_game_result # current game state after the moving
            
            template_reflection_assist = sokoban_reflection_template(sokoban_game_state=current_state_map, sokoban_new_game_state = sokoban_game_state, new_state=True)
            messages = [HumanMessage(content=template_reflection_assist)]
        
            logger.info(f""" 📝 Agent_Running Intermediate Steps : {sokoban_game_solution} 🔍 """)
            
            if "LEVEL_COMPLETED" in str(sokoban_game_result):
                LEVEL_COMPLETED = True
            
        return { "answers": sokoban_game_solution, "content": f"{content}", "role": "assistant",}

        
    async def sokoban_react_agent(self, user_query:str , model_name:str) -> dict: 
        """
        Docstring for sokoban_agent
        
        :param user_query: Description
        :param model_name: Description
        :param prompt_type: Description
        :return: Description
        """

        LEVEL_COMPLETED = False
        agent_scratchpad = []
        tools = [_makePlayerMove]
        sokoban_game_solution = []
        # tools = [isWall, isBlocked, isSokobanGameFinished, makePlayerMove]
        
        tool_names = f",".join([tool.name for tool in tools])
        content = "Your task is to solve the sokoban game"
        task_solve = f"Solve the given sokoban game"
        tool_text = render_text_description(tools)
        
        template_system = sokoban_system_template(user_query=user_query, tools=tool_text)
        template_assist = sokoban_assist_template(user_query=user_query, input = task_solve, tools = tool_text, tool_names=tool_names, agent_scratchpad=agent_scratchpad)
        
        generation_llm = ChatOllama(name = "Sokoban-Assistant-Agent", 
                                    model = model_name, 
                                    max_iterations = 1,
                                    callbacks = [AgentCallbackHandler()], 
                                    stop=["\nObservation", 
                                        "\nObservation:", 
                                        "Observation:", 
                                        "Observation"],
                                    temperature=0.5)
        
        if "llama3" not in model_name: # ollama3 does not allow banding tool
            llm_bind_tools = generation_llm.bind_tools(tools)
        else:
            llm_bind_tools = generation_llm
        
        generation_chain = llm_bind_tools | ReActSingleInputOutputParser()
        # messages = [SystemMessage(content = template_system), HumanMessage(content=template_assist)]
        messages = [HumanMessage(content=template_assist)]
        
        answers = ""
        agent_step = ""
        intermediate_steps = []
        
        try:
            
            while (not isinstance(agent_step, AgentFinish)) and (not LEVEL_COMPLETED):
                agent_step: Union[AgentAction, AgentFinish] = await generation_chain.ainvoke(messages)
                tool_calls = getattr(agent_step, "tool_calls", None) or []
                
                logger.info(f""" 🔬 Agent_Step : \n Moving: {''.join(sokoban_game_solution)} \n  Bool: AgentAction: {isinstance(agent_step, AgentAction)} | AgentFinish: {isinstance(agent_step, AgentFinish)} \n Tool_calls: {tool_calls} \n Agent_step: {agent_step} \n """)
                
                # If the model decides to call tools, execute them and return results
                if isinstance(agent_step, AgentAction) or len(tool_calls) > 0:
                    
                    if len(tool_calls) > 0:
                        for tool_call in tool_calls:
                            tool_name = tool_call.get("name")
                            tool_args = tool_call.get("args", {})
                            tool_call_id = tool_call.get("id")
                            
                            tool_to_use = self.find_tool_by_name(tool_list=tools, tool_name = tool_name)
                            if isinstance(tool_to_use, BaseTool):
                                Observation = tool_to_use.invoke(tool_args)
                            else:
                                Observation = f"Tool Called is Invalid '{tool_name}', use these tools {tool_names}"
                            intermediate_steps.append(ToolMessage(content=str(Observation), tool_call_id=tool_call_id))
                    else:
                        tool_name = agent_step.tool
                        tool_to_use = self.find_tool_by_name(tool_list=tools, tool_name = tool_name)
                        if isinstance(tool_to_use, BaseTool): # StructuredTool
                            tool_input = agent_step.tool_input
                            Observation = tool_to_use.func(str(tool_input))
                        else:
                            Observation = f"Tool Called is Invalid '{tool_name}', use these tools {tool_names}"
                        intermediate_steps.append((agent_step, str(Observation)))
                    
                    user_query = convert_current_state_to_map()
                    template_assist = sokoban_assist_template(user_query=user_query, input = task_solve, tools = tool_text, tool_names=tool_names, agent_scratchpad=agent_scratchpad)
                    # messages = [SystemMessage(content = template_system), HumanMessage(content=template_assist)]
                    messages = [HumanMessage(content=template_assist)]
                    logger.info(f""" 📝 Agent_Running Intermediate Steps Observation : {Observation} 🔍 """)
                    
                    if "VALID_MOVE" in str(Observation):
                        sokoban_game_solution.append(str(tool_input))
                    
                    if "LEVEL_COMPLETED" in str(Observation):
                        LEVEL_COMPLETED = True
                
                if isinstance(agent_step, AgentFinish) or LEVEL_COMPLETED:
                    if isinstance(agent_step, AgentFinish):
                        answers = agent_step.return_values
                    else:
                        answers = sokoban_game_solution
                    logger.info(f""" 📝 Agent_Running Final Answer is : {answers} ✅ """)
            
            return { 
                "answers": answers,
                "content": f"{content}",
                "role": "assistant",
                }
        
        except OutputParserException as e:
            logger.info(f""" 📝 Agent_Running OutputParserException | llm_output: {e.llm_output} | observation: {e.observation} ✅ """)
            return { 
                "answers": e.llm_output,
                "content": f"{content}",
                "role": "assistant",
                }
        except Exception as e:
            logger.error(f"❌ Agent Running failed: Type: {type(e)} | {e}")
            return {"answers":sokoban_game_solution, "content": f"{content}", "role": "assistant",} 
    
    def reflection_processing_moves(self, response, sokoban_game_solution) -> str:
        valid_steps = ""
        moving_steps = ""
        plan = response.strip().split('\n') 
        for direction in plan:
            if "<U>" in direction or "(U)" in direction or "UP" in direction or "U" in direction or "'U'" in direction or "**U**" in direction: 
                processed_move = makePlayerMove('U')
                if "VALID_MOVE" in str(processed_move):
                    valid_steps += "U"
                moving_steps += direction +" | Move result: "+processed_move+ "\n"
                
            if "<D>" in direction or "(D)" in direction or "DOWN" in direction or "D" in direction or "'D'" in direction or "**D**" in direction:
                processed_move = makePlayerMove('D')
                if "VALID_MOVE" in str(processed_move):
                    valid_steps += "D"
                moving_steps += direction +" | Move result: "+processed_move+ "\n"
                
            if "<L>" in direction or "(L)" in direction or "LEFT" in direction or "L" in direction or "'L'" in direction or "**L**" in direction: 
                processed_move = makePlayerMove('L')
                if "VALID_MOVE" in str(processed_move):
                    valid_steps += "L"
                moving_steps += direction +" | Move result: "+processed_move+ "\n"
                
            if "<R>" in direction or "(R)" in direction or "RIGHT" in direction or "R" in direction or "'R'" in direction or "**R**" in direction:
                processed_move = makePlayerMove('R')
                if "VALID_MOVE" in str(processed_move):
                    valid_steps += "R"
                moving_steps += direction +" | Move result: "+processed_move+ "\n"
                
        sokoban_game_solution.append(valid_steps)
        return moving_steps
    
    def post_processing_moves(self, response) -> str:
        steps = ""
        plan = response.strip().split('\n') 

        for direction in plan:
            if "<U>" in direction or "(U)" in direction or "UP" in direction or "U" in direction or "'U'" in direction or "**U**" in direction: 
                steps += "U"
            if "<D>" in direction or "(D)" in direction or "DOWN" in direction or "D" in direction or "'D'" in direction or "**D**" in direction:
                steps += "D" 
            if "<L>" in direction or "(L)" in direction or "LEFT" in direction or "L" in direction or "'L'" in direction or "**L**" in direction: 
                steps += "L"
            if "<R>" in direction or "(R)" in direction or "RIGHT" in direction or "R" in direction or "'R'" in direction or "**R**" in direction:
                steps += "R"
        return steps

    def find_tool_by_name(self, tool_name, tool_list: List[BaseTool]):
        # some time the tool name came like this 'makePlayerMove('D')'
        if '(' in tool_name or ')' in tool_name:
            tool_name = tool_name.split('(')[0]
            
        for tool in tool_list:
            if tool.name == tool_name:
                return tool
        raise f"Invalid_Tool"