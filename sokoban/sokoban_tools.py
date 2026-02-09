import os
import imageio
import numpy as np
import logging
# import pkg_resources
from typing import List
from PIL import Image, ImageDraw

logger = logging.getLogger("Sokoban-Agentic-Moving (SAM)")

def map_sign_to_suface_id(sign: str) -> int:
    """
    Maps a character sign to a surface ID.
    wall, floor, box_target, box_on_target, box, player, player_on_target
    """
    if sign == '#':
        return 0  # wall
    elif sign == ' ':
        return 1  # empty space
    elif sign == '.':
        return 2  # box target
    elif sign == '*':
        return 3  # box on target
    elif sign == '$':
        return 4  # box not on target
    elif sign == '@':
        return 5  # player
    elif sign == '+':
        return 6  # player on target
    else:
        print("sign:", sign)
        raise ValueError(f"Unknown sign: {sign}")

def room_to_img(serialized_map: str):
    pass

# def room_to_img(serialized_map: str):
#     """
#     Creates an RGB image of the room.
#     :param room:
#     :param room_structure:
#     :return:
#     """
#     resource_package = __name__
#     room = serialized_map.split("\n")
#     room = [row.strip() for row in room]
#     height = len(room)
#     width = len(room[0])
    
#     # Load images, representing the corresponding situation
#     box_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'box.png')))
#     box = imageio.imread(box_filename)

#     box_on_target_filename = pkg_resources.resource_filename(resource_package,
#                                                              '/'.join(('surface', 'box_on_target.png')))
#     box_on_target = imageio.imread(box_on_target_filename)

#     box_target_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'box_target.png')))
#     box_target = imageio.imread(box_target_filename)

#     floor_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'floor.png')))
#     floor = imageio.imread(floor_filename)

#     player_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'player.png')))
#     player = imageio.imread(player_filename)

#     player_on_target_filename = pkg_resources.resource_filename(resource_package,
#                                                                 '/'.join(('surface', 'player_on_target.png')))
#     player_on_target = imageio.imread(player_on_target_filename)

#     wall_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'wall.png')))
#     wall = imageio.imread(wall_filename)

#     surfaces = [wall, floor, box_target, box_on_target, box, player, player_on_target]

#     # Assemble the new rgb_room, with all loaded images
#     room_rgb = np.zeros(shape=(height * 16, width * 16, 3), dtype=np.uint8)
#     for i in range(height):
#         x_i = i * 16

#         for j in range(width):
#             y_j = j * 16
#             surface_sign = room[i][j]
#             surfaces_id = map_sign_to_suface_id(surface_sign)
#             room_rgb[x_i:(x_i + 16), y_j:(y_j + 16), :] = surfaces[surfaces_id]

#     return room_rgb

def create_gif(observations, filename="sokoban.gif", success=False, logger = None):
    """Create GIF from observations"""
    if not observations:
        print("No observations to visualize")
        return

    images = []
    total_frames = len(observations)
    for i in range(total_frames):

        # Get observation and ensure correct format
        obs = observations[i]

        # Create PIL image
        img = Image.fromarray(room_to_img(obs))

        if i == total_frames - 1:
            Im = ImageDraw.Draw(img)
            if success:
                Im.text((20, 20), "Success!",fill=(255, 255, 255))
        
        images.append(img)


    # Save as GIF
    durations = [200] * (len(images) - 1) + [2000]  # last frame stays longer (1s)
    if images:
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0
        )
        if logger:
            logger.info(f"GIF saved to {filename}")
        else:
            print(f"GIF saved to {filename}")
    else:
        if logger:
            logger.info(f"No images to create GIF")
        else:
            print("No images to create GIF")
        
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

    def isWall(self, x, y):
        """Returns True if the (x, y) position on
        the map is a wall, otherwise return False."""
        if x < 0 or x >= len(self.mapObj) or y < 0 or y >= len(self.mapObj[x]):
            return False # x and y aren't actually on the map.
        elif self.mapObj[x][y] in ('#', 'x'):
            return True # wall is blocking
        return False

    def isBlocked(self, x, y):
        """Returns True if the (x, y) position on the map is
        blocked by a wall or star, otherwise return False."""

        if self.isWall(x, y):
            return True

        elif x < 0 or x >= len(self.mapObj) or y < 0 or y >= len(self.mapObj[x]):
            return True # x and y aren't actually on the map.

        elif (x, y) in self.gameStateObj['boxes']:
            return True # a box is blocking
        return False

    def makeMove(self, playerMoveTo):
        """Given a map and game state object, see if it is possible for the
        player to make the given move. If it is, then change the player's
        position (and the position of any pushed box). If not, do nothing.
        Returns True if the player moved, otherwise False."""

        # Make sure the player can move in the direction they want.
        playerx, playery = self.gameStateObj['player']
        boxes = self.gameStateObj['boxes']

        if str(playerMoveTo).lower() == "U".lower():
            xOffset = -1
            yOffset = 0
        elif str(playerMoveTo).lower() == "R".lower():
            xOffset = 0
            yOffset = 1
        elif str(playerMoveTo).lower() == "D".lower():
            xOffset = 1
            yOffset = 0
        elif str(playerMoveTo).lower() == "L".lower():
            xOffset = 0
            yOffset = -1

        # See if the player can move in that direction.
        if self.isWall(playerx + xOffset, playery + yOffset):
            logger.info(f""" 🔀 makeMOVE: isWALL move {playerMoveTo} """)
            return False
        else:
            if (playerx + xOffset, playery + yOffset) in boxes:
                if not self.isBlocked(playerx + (xOffset*2), playery + (yOffset*2)):
                    ind = boxes.index((playerx + xOffset, playery + yOffset))
                    boxes[ind] = (boxes[ind][0] + xOffset, boxes[ind][1] + yOffset)
                else:
                    logger.info(f""" 🔀 makeMOVE: isBlocked move {playerMoveTo} """)
                    return False
            self.gameStateObj['player'] = (playerx + xOffset, playery + yOffset)
            return True

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

    # store back the current map conditions to file again for rendering/observation purposing 
    def description_map_state(self) -> str:
        player = f"\n Player (@) current position: {self.player}"
        box = f"\n Box ($) current positions: {self.boxes}"
        target = f"\n Target (.) current positions: {self.targets}"
        
        return f"{player} {box} {target}"
        
    def convert_current_state_to_map(self) -> List[List[str]]:
        map_width = self.levelObj['width']
        map_height = self.levelObj['height']
        map = self.levelObj['mapObj']

        starting_map = self.levelObj['mapObj']
        goal_pos = self.levelObj['goals']
        
        for i in range(map_height):
            for j in range(map_width):
                if starting_map[i][j] == "#":
                    map[i][j] = "#"
                # updated player and stars position
                elif starting_map[i][j] == ".":
                    if (i, j) == self.gameStateObj['player']:
                        map[i][j] = "+"
                    elif (i, j) in self.gameStateObj['boxes']:
                        map[i][j] = "*"
                    else:
                        map[i][j] = "."
                elif (i, j) == self.gameStateObj['player']:
                        map[i][j] = "@"
                elif (i, j) in self.gameStateObj['boxes']:
                        map[i][j] = "$"
                else:
                # other places are just floor
                    map[i][j] = " "
        return map

    def serialize_map(self) -> str:
        return "\n".join("".join(row) for row in self.convert_current_state_to_map())

    def deserialize_map(self, s: str) -> List[List[str]]:
        return [list(row) for row in s.splitlines()]

    def isLevelFinished(self):
        """Returns True if all the goals have stars in them."""
        for goal in self.levelObj['goals']:
            if goal not in self.gameStateObj['boxes']:
                return goal
        return True

global global_sokobanGame

global_sokobanGame = SokobanRules(os.path.join(os.getcwd(), "dataset/test/4_1.txt"))