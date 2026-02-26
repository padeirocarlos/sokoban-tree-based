import re
import copy
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from .instructions import sokoban_reflection_template
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv(override=True)
logger = logging.getLogger("Sokoban-Agentic-Workflow")


def parse_direction(text: str) -> str | None:
    """Parse a single direction (U/D/L/R) from a line of text.

    Checks for explicit markers like <U>, (U), **U** first,
    then falls back to keyword matching (UP/DOWN/LEFT/RIGHT)
    and finally bare letter matching with word-boundary awareness.
    Returns None if no direction is found.
    """
    text_upper = text.upper()

    # Priority 1: explicit delimited markers
    marker_match = re.search(r'<([UDLR])>', text_upper)
    if marker_match:
        return marker_match.group(1)

    paren_match = re.search(r'\(([UDLR])\)', text_upper)
    if paren_match:
        return paren_match.group(1)

    bold_match = re.search(r'\*\*([UDLR])\*\*', text_upper)
    if bold_match:
        return bold_match.group(1)

    quote_match = re.search(r"'([UDLR])'", text_upper)
    if quote_match:
        return quote_match.group(1)

    # Priority 2: full direction words
    for word, letter in [("UP", "U"), ("DOWN", "D"), ("LEFT", "L"), ("RIGHT", "R")]:
        if re.search(r'\b' + word + r'\b', text_upper):
            return letter

    # Priority 3: bare letter with word boundary
    for letter in ["U", "D", "L", "R"]:
        if re.search(r'\b' + letter + r'\b', text_upper):
            return letter

    return None


def convert_current_state_to_map(sokoban_game) -> str:
    map_width = sokoban_game.level['width']
    map_height = sokoban_game.level['height']
    game_map = copy.deepcopy(sokoban_game.level['map_data'])
    starting_map = sokoban_game.level['map_data']

    for i in range(map_height):
        for j in range(map_width):
            if starting_map[i][j] == "#":
                game_map[i][j] = "#"
            elif starting_map[i][j] == ".":
                if (i, j) == sokoban_game.game_state['player']:
                    game_map[i][j] = "+"
                elif (i, j) in sokoban_game.game_state['boxes']:
                    game_map[i][j] = "*"
                else:
                    game_map[i][j] = "."
            elif (i, j) == sokoban_game.game_state['player']:
                    game_map[i][j] = "@"
            elif (i, j) in sokoban_game.game_state['boxes']:
                    game_map[i][j] = "$"
            else:
                game_map[i][j] = " "

    game_map_str = "\n".join("".join(row) for row in game_map)
    player = f"\n Player (@) position: {sokoban_game.game_state['player']}"
    box = f"\n Box ($) positions: {sorted(sokoban_game.game_state['boxes'])}"
    target = f"\n Target (.) positions: {sokoban_game.targets}"

    return f"{game_map_str} \n {player} {box} {target}"


def make_player_move(player_moving: str, sokoban_game) -> str:
    """Attempt to move the player in a specified direction in the Sokoban game."""

    def is_wall(x, y):
        if x < 0 or x >= len(sokoban_game.map_data) or y < 0 or y >= len(sokoban_game.map_data[x]):
            return True
        elif sokoban_game.map_data[x][y] in ('#', 'x'):
            return True
        return False

    def is_blocked(x, y):
        if is_wall(x, y):
            return True
        elif x < 0 or x >= len(sokoban_game.map_data) or y < 0 or y >= len(sokoban_game.map_data[x]):
            return True
        elif (x, y) in sokoban_game.game_state['boxes']:
            return True
        return False

    def is_level_finished():
        for goal in sokoban_game.level['goals']:
            if goal not in sokoban_game.game_state['boxes']:
                return goal
        return True

    playerx, playery = sokoban_game.game_state['player']
    boxes = sokoban_game.game_state['boxes']

    level_check = is_level_finished()
    if isinstance(level_check, bool):
        return "LEVEL_COMPLETED"

    parsed = parse_direction(str(player_moving))
    if parsed is None:
        return f"Invalid direction: '{player_moving}'. Use U/D/L/R or UP/DOWN/LEFT/RIGHT"

    offsets = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
    x_offset, y_offset = offsets[parsed]

    target_x = playerx + x_offset
    target_y = playery + y_offset

    if is_wall(target_x, target_y):
        return f"Cannot move, because the player's new position ({target_x}, {target_y}) is a wall, try a different move"

    if (target_x, target_y) in boxes:
        box_push_x = target_x + x_offset
        box_push_y = target_y + y_offset

        if not is_blocked(box_push_x, box_push_y):
            boxes.remove((target_x, target_y))
            boxes.add((box_push_x, box_push_y))
            sokoban_game.game_state['boxes'] = boxes
            sokoban_game.game_state['player'] = (target_x, target_y)
            box_str = ','.join([str((box[0], box[1])) for box in sorted(boxes)])
            return f"It is VALID_MOVE, the Player's new position is ({target_x}, {target_y}) \n the box's new position {box_str} \n"
        else:
            return f"Cannot move, because the box's new position ({box_push_x}, {box_push_y}) is blocked, try a different move"

    sokoban_game.game_state['player'] = (target_x, target_y)
    return f"It is VALID_MOVE, the player's new position is ({target_x}, {target_y}) \n"


class AgentCallbackHandler(BaseCallbackHandler):

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        logger.info(f"Agent_Calling: Prompt to LLM \n {prompts[0]} \n")


class SokobanAgentic:

    def __init__(self, model_name: str="llama3"):
        self.model_name = model_name

    async def sokoban_reflection_agent(self, sokoban_game: str, model_name: str, sokoban_rules) -> dict:
        iterations = 0
        MAX_ITERATIONS = 5
        LEVEL_COMPLETED = False
        sokoban_game_solution = []
        content = "Your task is to solve the sokoban game"

        template_reflection_assist = sokoban_reflection_template(sokoban_game_state=sokoban_game)

        generation_llm = ChatOllama(name="Sokoban-Assistant-Agent",
                                    model=model_name,
                                    callbacks=[AgentCallbackHandler()],
                                    max_iterations=1,
                                    temperature=0.5)

        generation_chain = generation_llm
        messages = [HumanMessage(content=template_reflection_assist)]

        while not LEVEL_COMPLETED and MAX_ITERATIONS >= iterations:
            result = await generation_chain.ainvoke(messages)

            current_state_map = convert_current_state_to_map(sokoban_rules)
            sokoban_game_result = self.reflection_processing_moves(result.content, sokoban_game_solution, sokoban_rules)
            sokoban_game_state = convert_current_state_to_map(sokoban_rules) + "\n" + sokoban_game_result

            template_reflection_assist = sokoban_reflection_template(sokoban_game_state=current_state_map, sokoban_new_game_state=sokoban_game_state, new_state=True)
            messages = [HumanMessage(content=template_reflection_assist)]

            if "LEVEL_COMPLETED" in str(sokoban_game_result):
                LEVEL_COMPLETED = True
            iterations += 1

        return {"answers": sokoban_game_solution, "content": content, "role": "assistant"}

    def reflection_processing_moves(self, response, sokoban_game_solution, sokoban_rules) -> str:
        valid_steps = ""
        moving_steps = ""
        plan = response.strip().split('\n')
        for line in plan:
            parsed = parse_direction(line)
            if parsed is not None:
                processed_move = make_player_move(parsed, sokoban_rules)
                if "VALID_MOVE" in str(processed_move):
                    valid_steps += parsed
                moving_steps += line + " | Move result: " + processed_move + "\n"

        sokoban_game_solution.append(valid_steps)
        return moving_steps

    def post_processing_moves(self, response) -> str:
        steps = ""
        plan = response.strip().split('\n')
        for line in plan:
            parsed = parse_direction(line)
            if parsed is not None:
                steps += parsed
        return steps

    def find_tool_by_name(self, tool_name, tool_list: list):
        if '(' in tool_name or ')' in tool_name:
            tool_name = tool_name.split('(')[0]

        for tool in tool_list:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Invalid tool: {tool_name}")
