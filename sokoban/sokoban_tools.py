import logging

logger = logging.getLogger("Sokoban-Agentic-Moving (SAM)")


class SokobanRules:
    """Sokoban game engine: parses map files and tracks game state."""

    def __init__(self, file_path):
        self.map_data = None
        self.targets = None
        self.level = None
        self.game_state = None
        self._read_map(file_path)

    def _read_map(self, file_path):
        current_map = []
        with open(file_path, 'r') as sf:
            for line in sf.readlines():
                if line.startswith('#'):
                    current_map.append(line.strip())
                else:
                    break

        self.map_data = [list(mapline) for mapline in current_map]

        startx = None
        starty = None
        goals = []
        boxes = set()

        for x in range(len(self.map_data)):
            for y in range(len(self.map_data[0])):
                cell = self.map_data[x][y]
                if cell == '@':
                    startx = x
                    starty = y
                elif cell == '.':
                    goals.append((x, y))
                elif cell == '$':
                    boxes.add((x, y))
                elif cell == '*':
                    goals.append((x, y))
                    boxes.add((x, y))

        assert startx is not None and starty is not None, (
            'Level missing a "@" or "+" to mark the start point.'
        )
        assert len(goals) > 0, 'Level must have at least one goal.'
        assert len(boxes) >= len(goals), (
            f'Level is impossible to solve. It has {len(goals)} goals but only {len(boxes)} boxes.'
        )

        self.targets = ",".join([f"({goal[0]}, {goal[1]})" for goal in goals])

        self.game_state = {
            'player': (startx, starty),
            'boxes': boxes,
        }

        self.level = {
            'width': len(self.map_data[0]),
            'height': len(self.map_data),
            'map_data': self.map_data,
            'goals': goals,
        }
