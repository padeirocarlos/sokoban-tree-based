import os
import logging

logger = logging.getLogger("Sokoban-Agentic-Moving (SAM)")


class SokobanRules:
    """A class that contains the AI for the Sokoban rules."""

    def __init__(self, data_file=None):
        self.map_data = None
        self.player = None
        self.boxes = None
        self.targets = None
        self.level = None
        self.game_state = None
        self.DATA_FILE = data_file
        self.ACTION_SEQUENCE = None
        self.read_map(data_file)
        self.data_file = str(data_file).rsplit("/", 1)[1]

    def read_map(self, file_path):
        current_map = []
        with open(file_path, 'r') as sf:
            for line in sf.readlines():
                if '#' == line[0]:
                    current_map.append(line.strip())
                else:
                    break

        self.map_data = [list(mapline) for mapline in current_map]

        startx = None
        starty = None
        goals = []
        boxes = set()
        obstacles = []

        for x in range(len(self.map_data)):
            for y in range(len(self.map_data[0])):
                if self.map_data[x][y] == '@':
                    startx = x
                    starty = y
                if self.map_data[x][y] == '.':
                    goals.append((x, y))
                if self.map_data[x][y] == '$':
                    boxes.add((x, y))
                if self.map_data[x][y] == '#':
                    obstacles.append((x, y))
                if self.map_data[x][y] == "*":
                    goals.append((x, y))
                    boxes.add((x, y))

        assert startx is not None and starty is not None, 'Level missing a "@" or "+" to mark the start point.'
        assert len(goals) > 0, 'Level must have at least one goal.'
        assert len(boxes) >= len(goals), 'Level is impossible to solve. It has %s goals but only %s stars.'

        self.player = f"({startx},{starty})"
        self.boxes = ",".join([f"({box[0]}, {box[1]})" for box in sorted(boxes)])
        self.targets = ",".join([f"({goal[0]}, {goal[1]})" for goal in goals])

        self.game_state = {
            'player': (startx, starty),
            'stepCounter': 0,
            'obstacles': obstacles,
            'boxes': boxes,
        }

        self.level = {
            'width': len(self.map_data[0]),
            'height': len(self.map_data),
            'map_data': self.map_data,
            'goals': goals,
            'startState': self.game_state,
        }
