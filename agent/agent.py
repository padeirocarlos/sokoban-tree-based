import re
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from .instructions import sokoban_reflection_template

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

    # Priority 1: explicit delimited markers — <U>, (U), **U**, 'U'
    marker_match = re.search(r"<([UDLR])>|\(([UDLR])\)|\*\*([UDLR])\*\*|'([UDLR])'", text_upper)
    if marker_match:
        return next(g for g in marker_match.groups() if g is not None)

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
    """Render the current game state as a Sokoban map string with annotations."""
    height = sokoban_game.level['height']
    width = sokoban_game.level['width']
    base_map = sokoban_game.level['map_data']
    player = sokoban_game.game_state['player']
    boxes = sokoban_game.game_state['boxes']

    game_map = []
    for i in range(height):
        row = []
        for j in range(width):
            is_goal = base_map[i][j] in ('.', '*')
            is_player = (i, j) == player
            is_box = (i, j) in boxes

            if base_map[i][j] == '#':
                row.append('#')
            elif is_player:
                row.append('+' if is_goal else '@')
            elif is_box:
                row.append('*' if is_goal else '$')
            elif is_goal:
                row.append('.')
            else:
                row.append(' ')
        game_map.append(row)

    map_str = "\n".join("".join(row) for row in game_map)
    player_info = f"\n Player (@) position: {player}"
    box_info = f"\n Box ($) positions: {sorted(boxes)}"
    target_info = f"\n Target (.) positions: {sokoban_game.targets}"

    return f"{map_str} \n {player_info} {box_info} {target_info}"


def make_player_move(player_moving: str, sokoban_game) -> str:
    """Attempt to move the player in a specified direction in the Sokoban game."""

    def is_wall(x, y):
        if x < 0 or x >= len(sokoban_game.map_data) or y < 0 or y >= len(sokoban_game.map_data[x]):
            return True
        return sokoban_game.map_data[x][y] in ('#', 'x')

    def is_blocked(x, y):
        return is_wall(x, y) or (x, y) in sokoban_game.game_state['boxes']

    def is_level_finished():
        return all(goal in sokoban_game.game_state['boxes'] for goal in sokoban_game.level['goals'])

    playerx, playery = sokoban_game.game_state['player']
    boxes = sokoban_game.game_state['boxes']

    if is_level_finished():
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

    MAX_REFLECTION_ITERATIONS = 5

    async def sokoban_reflection_agent(self, sokoban_game: str, model_name: str, sokoban_rules) -> dict:
        """Run iterative LLM reflection to solve the Sokoban puzzle."""
        sokoban_game_solution = []

        llm = ChatOllama(
            name="Sokoban-Assistant-Agent",
            model=model_name,
            callbacks=[AgentCallbackHandler()],
            max_iterations=1,
            temperature=0.5,
        )

        prompt = sokoban_reflection_template(sokoban_game_state=sokoban_game)
        messages = [HumanMessage(content=prompt)]
        level_completed = False

        for _ in range(self.MAX_REFLECTION_ITERATIONS):
            if level_completed:
                break

            result = await llm.ainvoke(messages)

            state_before = convert_current_state_to_map(sokoban_rules)
            move_results = self._process_moves(result.content, sokoban_game_solution, sokoban_rules)
            state_after = convert_current_state_to_map(sokoban_rules) + "\n" + move_results

            prompt = sokoban_reflection_template(
                sokoban_game_state=state_before,
                sokoban_new_game_state=state_after,
                new_state=True,
            )
            messages = [HumanMessage(content=prompt)]

            if "LEVEL_COMPLETED" in move_results:
                level_completed = True

        return {"answers": sokoban_game_solution, "role": "assistant"}

    def _process_moves(self, response, sokoban_game_solution, sokoban_rules) -> str:
        """Parse LLM response lines, execute valid moves, and track results."""
        valid_steps = ""
        moving_steps = ""
        for line in response.strip().split('\n'):
            parsed = parse_direction(line)
            if parsed is not None:
                processed_move = make_player_move(parsed, sokoban_rules)
                if "VALID_MOVE" in processed_move:
                    valid_steps += parsed
                moving_steps += f"{line} | Move result: {processed_move}\n"

        sokoban_game_solution.append(valid_steps)
        return moving_steps
