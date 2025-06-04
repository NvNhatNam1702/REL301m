import heapq
import numpy as np
from typing import List, Tuple, Set, Dict

class AStarAgent:
    def __init__(self, env):
        self.env = env
        self.grid = env.grid
        self.grid_height = env.grid_height
        self.grid_width = env.grid_width
        self.path = []
        self.current_step = 0

    def heuristic(self, a, b):
        """Calculate the Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        """Get valid neighboring positions."""
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
            new_r, new_c = pos[0] + dr, pos[1] + dc
            if (0 <= new_r < self.grid_height and 
                0 <= new_c < self.grid_width and 
                (self.grid[new_r, new_c] == 0 or (new_r, new_c) == self.env.target_pod)):
                neighbors.append((new_r, new_c))
        return neighbors

    def find_path(self, start, goal):
        """Find path using A* algorithm."""
        # Initialize data structures
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current = heapq.heappop(frontier)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for next_pos in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        return None  # No path found

    def find_complete_path(self, start_pos):
        """Find path from start to target pod, then to delivery location."""
        # First find path to target pod
        pod_path = self.find_path(start_pos, self.env.target_pod)
        if not pod_path:
            return None

        # Then find path from target pod to delivery location
        delivery_path = self.find_path(self.env.target_pod, self.env.delivery_locations[0])
        if not delivery_path:
            return None

        # Combine paths (excluding the target pod from the second path to avoid duplication)
        return pod_path + delivery_path[1:]

    def get_next_action(self) -> Tuple[int, int]:
        """Get the next position in the path."""
        if self.current_step < len(self.path):
            next_pos = self.path[self.current_step]
            self.current_step += 1
            return next_pos
        return None

    def reset(self):
        """Reset the agent's state."""
        self.path = []
        self.current_step = 0 