"""Prompt templates for the Sokoban agentic workflow."""

SOKOBAN_RULES_TEXT = """
        Sokoban Rules:
        - Walls (#) are completely blocked
        - Treat walls as impassable boundaries
        - The player (@) cannot move into a wall
        - Boxes ($) cannot be pushed into a wall
        - Boxes ($) cannot be moved only can be pushed by player (@)

        Player (@) movement:
        - The player (@) can move up, down, left, or right into an empty space ( ) or a target (.)
        - The player (@) cannot move diagonally
        - The player (@) cannot move into a wall
        - The player (@) cannot move through boxes ($)
        - The player (@) cannot pull boxes (only push)

        Pushing boxes ($):
        - If the player (@) moves into a box, the box is pushed one step in the same direction;
        - A box ($) cannot be moved only can be pushed by player (@)
        - A box can only be pushed if the square behind it is free (empty or target .);
        - A box cannot be pushed into a wall or another box.

        Winning condition:
        - The game is solved when all boxes ($) are placed on all targets (.).

        Sokoban map format:
        - The map is stored in a nested list;
        - Each item represents the element at that position
        - '#' represents the wall
        - '@' represents player
        - '$' represents box
        - '.' represents target
        - ' ' represents the empty free space which player or box can move into.
"""

_MOVE_EXAMPLES = """
        Directions moves sokoban output and requirements example:

        1. <U> Move Up the player from (7,7) to (6,7), box positions are kept as (3,3),(5,5)
        2. <U> Move Up the player from (3,2) to (2,2), box positions are kept as (3,3),(7,6)
        3. <D> Move Down the player from (4,5) to (5,5), box positions are updated to (3,3), (6,5)
        4. <D> Move Down the player from (5,5) to (6,5), box positions are updated to (3,3), (7,5)
        4. <R> Move Right the player from (6,4) to (7,4), box positions are updated to (3,3) to (7,6)
        5. <R> Move Right the player from (2,2) to (2,3), box positions are kept as (3,3), (7,6)
        6. <L> Move Left the player from (4,7) to (4,6), box positions are kept as (3,3), (5,5)
        7. <L> Move Left the player from (4,6) to (4,5), box positions are kept as (3,3), (5,5)
"""


def sokoban_reflection_template(
    sokoban_game_state: str,
    sokoban_new_game_state: str = None,
    new_state: bool = False,
) -> str:
    """Build a prompt for the LLM to solve or reflect on a Sokoban puzzle."""
    baseline = f"""
        Context:
            The sokoban game is a puzzle game where the player (@) must push boxes ($) onto target locations in a grid-like environment.
            The player can only move in four directions: up <U>, down <D>, left <L>, right <R>, and cannot move through walls or other boxes.
            The goal is to push all boxes onto their respective target locations. The steps should be in the format of a numbered list,
            where each step describes the player's movement.
        {SOKOBAN_RULES_TEXT}
        {_MOVE_EXAMPLES}
    """

    if new_state:
        task_prompt = f"""
        You are an expert Sokoban player and strategist.
        Your task is to produce a clear, detailed, step-by-step plan to solve the Sokoban puzzle provided between triple backticks.

        You will be given:
        - The previous Sokoban game state and the solution that was attempted: \n ```{sokoban_game_state}``` \n;
        - The current Sokoban game state after executing that solution: \n ```{sokoban_new_game_state}``` \n

        Using this information, your goals are:
        1. Analyze the differences between the previous and current game states.
        2. Identify any mistakes, inefficiencies, or unexpected outcomes in the previous solution.
        3. Provide an improved, corrected, and optimized step-by-step solution to continue solving the puzzle from the current game state.
        4. Ensure the plan avoids deadlocks, unnecessary backtracking, and preserves solvability.
        """
    else:
        task_prompt = f"""
        You are a skilled player of Sokoban game.

        Task:
            Your task is to provide a detailed step-by-step plan to solve the sokoban game delimited by triple
            backticks:```{sokoban_game_state}```.
        """

    return f"{task_prompt} \n {baseline}"

