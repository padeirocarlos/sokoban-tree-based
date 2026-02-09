import os
from dotenv import load_dotenv
from datetime import datetime
load_dotenv(override=True)
dt = datetime.now()

def sokoban_system_template(user_query:str, tools:str) -> str:
    
    sokoban_prompt = f"""
        Task:
        Your task is to solve the sokoban game delimited by triple backticks: ```{user_query}```. You have access to the following tools:
            {tools}
        
        Context: 
        The sokoban game is a puzzle game where the player (@) must push boxes ($) onto target locations in a grid-like environment. 
        The player can only move in four directions: up <U>, down <D>, left <L>, right <R>, and cannot move through walls or other boxes. 
        The goal is to push all boxes onto their respective target locations.
        
        Sokoban Rules:
        - Walls (#) are completely blocked
        - The player (@) cannot move into a wall
        - Boxes ($) cannot be pushed into a wall
        - Boxes ($) cannot be moved only can be pushed by player (@)
        - Treat walls as impassable boundaries.

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
        - "#" represents the wall
        - "@" represents player
        - "$" represents box
        - "." represents target
        - " " represents the empty free space which player or box can move into. 
    """
    return sokoban_prompt

def sokoban_assist_template(user_query:str, input:str, tools:str, tool_names:str, agent_scratchpad:list) -> str:
    
    sokoban_react_prompt = f"""
        Task:
            Solve the sokoban game delimited by triple backticks:\n```{user_query}```\n. You have access to the following tools:
            {tools}
            Your task is to solve the sokoban game delimited by triple backticks: \n```{user_query}```\n.
            
        Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            CRITICAL: You MUST end with "Final Answer:" when you know the answer.
            
            Begin!
            
            Question: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            Thought: {agent_scratchpad}
    """
    return sokoban_react_prompt
    
def sokoban_reflection_template(sokoban_game_state:str, sokoban_new_game_state:str=None, new_state:bool=False) -> str:
      
    sokoban_baseline_prompt = f"""
        Context: 
            The sokoban game is a puzzle game where the player (@) must push boxes ($) onto target locations in a grid-like environment. 
            The player can only move in four directions: up <U>, down <D>, left <L>, right <R>, and cannot move through walls or other boxes. 
            The goal is to push all boxes onto their respective target locations. The steps should be in the format of a numbered list, 
            where each step describes the player's movement.
        
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
        - '$' represents box
        - '.' represents target
        - '@' represents player
        - '#' represents the wall
        - ' ' represents the empty free space which player or box can move into. 
        
        Sokoban output requirements example:
        
        1. <U> Move the player from (7,7) to (6,7), box positions are kept as (3,3),(5,5)
        2. <U> Move the player from (3,2) to (2,2), box positions are kept as (3,3),(7,6)
        3. <D> Move the player from (2,3) to (3,3), box positions are updated to (4,3) to (7,6)
        4. <R> Move the player from (6,4) to (7,4), box positions are updated to (3,3) to (7,6)
        5: <L> Move the player from (7,3) to (7,2), box positions are kept as (3,3), (7,6)
    """

    sokoban_normal_prompt = f"""
        You are a skilled player of Sokoban game. 
        
        Task:
            Your task is to provide a detailed step-by-step plan to solve the sokoban game delimited by triple 
            backticks:```{sokoban_game_state}```.
        """
        
    sokoban_reflect_prompt = f"""
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
    sokoban_prompt = f"{sokoban_reflect_prompt} \n {sokoban_baseline_prompt}" if new_state else f"{sokoban_normal_prompt} \n {sokoban_baseline_prompt}" 
    
    return sokoban_prompt

def ___sokoban_assist_template(user_query:str) -> str:
    
    sokoban_prompt = f"""
    
        You are a skilled player of Sokoban game. 
        
        Task:
        Your task is to provide a detailed step-by-step plan to solve the sokoban game delimited by triple 
        backticks: ```{user_query}```.
        
        Context: 
        The sokoban game is a puzzle game where the player must push boxes onto target locations in a grid-like environment. 
        The player can only move in four directions (up, down, left, right) and cannot move through walls or other boxes. 
        The goal is to push all boxes onto their respective target locations. The steps should be in the format of a numbered list, 
        where each step describes the player's movement and the box movement if applicable.
        
        Sokoban Rules:

        - Walls (#) are completely blocked.
        - The player cannot move into a wall.
        - Boxes cannot be pushed into a wall.
        - Treat walls as impassable boundaries.

        Player movement:
        
        - The player (@) can move up, down, left, or right into an empty space ( ) or a target (.).
        - The player cannot move diagonally.
        - The player cannot move into a wall.

        Pushing boxes ($):
        - If the player moves into a box, the box is pushed one step in the same direction.
        - A box can only be pushed if the square behind it is free (empty or target .).
        - A box cannot be pushed into a wall or another box.

        Winning condition:
        - The game is solved when all boxes ($) are placed on all targets (.).

        Sokoban map format:
        
        The map is stored in a nested list. Each item represents the element at that position. 
        "#" represents the wall, "@" represents player, "$" represents box, and "." represents target. 
        " " represents the empty free space which player or box can move into. 
        
        Example of sokoban map format:
        - MAP (indexing from 0):
            [['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'], 
            ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'], 
            ['#', '#', ' ', ' ', ' ', ' ', ' ', ' ', '#', '#'], 
            ['#', '#', ' ', '$', ' ', ' ', ' ', ' ', '#', '#'], 
            ['#', '#', ' ', ' ', ' ', ' ', ' ', ' ', '#', '#'], 
            ['#', '#', ' ', ' ', ' ', '$', ' ', ' ', '#', '#'], 
            ['#', '#', ' ', ' ', ' ', ' ', ' ', ' ', '#', '#'], 
            ['#', '#', ' ', '.', ' ', ' ', '.', '@', '#', '#'], 
            ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'], 
            ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#']]
        
        - Player (@) position: (7,7) 
        - Box ($) positions: (3,3), (5,5). 
        - Target (.) positions: (7,3), (7,6).
        
        Sokoban output requirements example:
        
        1. <U> Move the player from (7,7) to (6,7), box positions are kept as (3,3), (5,5)
        2. <U> Move the player from (6,7) to (5,7), box positions are kept as (3,3), (5,5)
        3. <U> Move the player from (5,7) to (4,7), box positions are kept as (3,3), (5,5)
        4. <L> Move the player from (4,7) to (4,6), box positions are kept as (3,3), (5,5)
        5. <L> Move the player from (4,6) to (4,5), box positions are kept as (3,3), (5,5)
        6. <D> Move the player from (4,5) to (5,5), box positions are updated to (3,3), (6,5)
        7. <D> Move the player from (5,5) to (6,5), box positions are updated to (3,3), (7,5)
        8. <L> Move the player from (6,5) to (6,4), box positions are kept as (3,3), (7,5)
        9. <D> Move the player from (6, 4) to (7, 4), box positions are kept as (3,3), (7,5)
        10. <R> Move the player from (6, 4) to (7, 4), box positions are updated to (3,3) to (7,6)
        11. <L> Move the player from (7,5) to (7,4), box positions are kept as (3,3), (7,6)
        12. <L> Move the player from (7,4) to (7,3), box positions are kept as (3,3), (7,6)
        13: <L> Move the player from (7,3) to (7,2), box positions are kept as (3,3), (7,6)
        14. <U> Move the player from (7,2) to (6,2), box positions are kept as (3,3), (7,6)
        15. <U> Move the player from (6,2) to (5,2), box positions are kept as (3,3), (7,6)
        16. <U> Move the player from (5,2) to (4,2), box positions are kept as (3,3), (7,6)        
        17. <U> Move the player from (4,2) to (3,2), box positions are kept as (3,3), (7,6)
        18. <U> Move the player from (3, 2) to (2,2), box positions are kept as (3,3), (7,6)
        19. <R> Move the player from (2,2) to (2,3), box positions are kept as (3,3), (7,6)
        20. <D> Move the player from (2,3) to (3,3), box positions are updated to (4,3) to (7,6)
        21. <D> Move the player from (3,3) to (4,3), box positions are updated to (5,3) to (7,6)
        22. <D> Move the player from (4,3) to (5,3), box positions are updated to (6,3) to (7,6)
        23. <D> Move the player from (5,3) to (6,3), box positions are updated to (7,3) to (7,6)
    """
    
    return sokoban_prompt
