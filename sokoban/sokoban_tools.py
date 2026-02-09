import os
import numpy as np
import logging
# import pkg_resources
from typing import List

logger = logging.getLogger("Sokoban-Agentic-Moving (SAM)")
        
class SokobanRules:
    """A class that contains the AI for the Sokoban rules."""

    def __init__(self, data_file=None):
        
        self.mapObj = None
        self.player = None
        self.boxes = None
        self.targets = None
        self.levelObj = None
        self.gameStateObj = None
        self.gameObstaclesObj = None
        self.DATA_FILE = data_file
        self.ACTION_SEQUENCE = None
        self.read_map(data_file)
        self.data_file = str(data_file).rsplit("/", 1)[1]

    def read_map(self, file_path):
        current_map = []
        with open(file_path, 'r') as sf:
            for line in sf.readlines():
                if '#' == line[0]: # if the current line contains # which represents wall, then continue add this line as current map
                    current_map.append(line.strip())
                else:
                    break  
                
        self.mapObj = [list(mapline) for mapline in current_map]     

        # Loop through the spaces in the map and find the @, ., and $
        # characters for the starting game state.
        startx = None # The x and y for the player's starting position
        starty = None
        goals = [] # list of (x, y) tuples for each goal.
        boxes = [] # list of (x, y) for each star's starting position.
        obstacles = [] # list of (x, y) for each obstacles position.

        for x in range(len(self.mapObj)):
            for y in range(len(self.mapObj[0])):

                if self.mapObj[x][y] == '@':
                    startx = x
                    starty = y

                if self.mapObj[x][y] == '.':
                    goals.append((x, y))
                if self.mapObj[x][y] == '$':
                    boxes.append((x, y))
                if self.mapObj[x][y] == '#':
                    obstacles.append((x, y))
                if self.mapObj[x][y] == "*":
                    goals.append((x, y))
                    boxes.append((x, y))
                 
        # Basic level design sanity checks:
        assert startx != None and starty != None, 'Level missing a "@" or "+" to mark the start point.' 
        assert len(goals) > 0, 'Levelmust have at least one goal.'
        assert len(boxes) >= len(goals), 'Level is impossible to solve. It has %s goals but only %s stars.'

        self.player = f"({startx},{starty})"
        self.boxes = f"{",".join([str(f"({box[0]}, {box[1]})") for box in boxes])}"
        self.targets = f"{",".join([str(f"({goal[0]}, {goal[1]})") for goal in goals])}"
        
        # Create level object and starting game state object.
        self.gameStateObj = {'player': (startx, starty),
                        'stepCounter': 0,
                        'obstacles': obstacles,
                        'boxes': boxes}
        
        self.levelObj = {'width': len(self.mapObj[0]),
                        'height': len(self.mapObj),
                        'mapObj': self.mapObj,
                        'goals': goals,
                        'startState': self.gameStateObj}

global global_sokobanGame
global_sokobanGame = SokobanRules(os.path.join(os.getcwd(), "dataset/test/1_4.txt"))