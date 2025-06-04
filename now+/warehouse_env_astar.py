import numpy as np
import tkinter as tk
from typing import List, Tuple, Set
import random

class WarehouseEnvAStar:
    def __init__(self):
        self.grid_height = 25
        self.grid_width = 25
        
        # Initialize grid with empty spaces
        self.grid = np.zeros((self.grid_height, self.grid_width))
        
        # Define workstations (starting positions)
        self.workstations = [
            (6, 2),
            (8, 2),
            (10, 2),
        ]
        
        # Define delivery location
        self.delivery_locations = [(15, 22)]
        
        # Randomly generate inventory pods within the specified area
        self.num_pods = 20
        self.pod_area_top = 3
        self.pod_area_left = 4
        self.pod_area_bottom = 19
        self.pod_area_right = 20
        self.inventory_pods = self._generate_random_pods()
        
        # Mark inventory pods as obstacles in the grid
        for pod in self.inventory_pods:
            self.grid[pod] = 1
        
        # Initialize visualization
        self.cell_size = 20
        self.window = None
        self.canvas = None
        self._init_visualization()
        
        # Initialize robot position at first workstation
        self.robot_pos = self.workstations[0]
        
        # Initialize target pod and delivered packages tracking
        self.available_pods = self.inventory_pods.copy()
        self.delivered_pods = set()
        self.target_pod = None
        self.select_new_target()
        
        # Initialize path
        self.path = []

    def _init_visualization(self):
        """Initialize the visualization window and canvas."""
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("Warehouse Environment - A* Pathfinding")
            self.canvas = tk.Canvas(
                self.window,
                width=self.grid_width * self.cell_size,
                height=self.grid_height * self.cell_size
            )
            self.canvas.pack()

    def _generate_random_pods(self):
        possible_positions = [
            (r, c)
            for r in range(self.pod_area_top, self.pod_area_bottom + 1)
            for c in range(self.pod_area_left, self.pod_area_right + 1)
            if (r, c) not in self.workstations and (r, c) not in self.delivery_locations
        ]
        return random.sample(possible_positions, self.num_pods)
        
    def select_new_target(self):
        """Select a new target pod from available pods."""
        if not self.available_pods:
            return False
        
        self.target_pod = random.choice(self.available_pods)
        self.available_pods.remove(self.target_pod)
        return True
        
    def mark_pod_delivered(self):
        """Mark the current target pod as delivered."""
        if self.target_pod:
            self.delivered_pods.add(self.target_pod)
            self.target_pod = None
            
    def all_pods_delivered(self):
        """Check if all pods have been delivered."""
        return len(self.delivered_pods) == len(self.inventory_pods)
        
    def render(self):
        """Render the warehouse environment."""
        self.canvas.delete("all")
        
        # Draw grid cells
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                if self.grid[r, c] == 1:  # Inventory pod
                    if (r, c) == self.target_pod:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="purple")
                    elif (r, c) in self.delivered_pods:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="gray")
                    else:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="green")
                elif (r, c) in self.workstations:  # Workstation
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="blue")
                elif (r, c) in self.delivery_locations:  # Delivery location
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="yellow")
                else:  # Empty space
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
        
        # Draw robot
        robot_x = self.robot_pos[1] * self.cell_size + self.cell_size // 2
        robot_y = self.robot_pos[0] * self.cell_size + self.cell_size // 2
        self.canvas.create_oval(
            robot_x - self.cell_size//3,
            robot_y - self.cell_size//3,
            robot_x + self.cell_size//3,
            robot_y + self.cell_size//3,
            fill="red"
        )
        
        # Draw path
        for pos in self.path:
            x = pos[1] * self.cell_size + self.cell_size // 2
            y = pos[0] * self.cell_size + self.cell_size // 2
            self.canvas.create_oval(
                x - self.cell_size//6,
                y - self.cell_size//6,
                x + self.cell_size//6,
                y + self.cell_size//6,
                fill="red"
            )
        
        self.window.update()
        
    def update_robot_position(self, new_pos):
        """Update robot position."""
        self.robot_pos = new_pos
        
    def clear_path(self):
        """Clear the current path."""
        self.path = []

    def reset(self):
        """Reset the environment to initial state."""
        if self.window:
            self.window.destroy()
        self.window = None
        self.canvas = None
        self._init_visualization()  # Reinitialize visualization
        
        self.robot_pos = self.workstations[0]
        self.path = []
        self.target_pod = None
        # Regenerate pods randomly
        self.inventory_pods = self._generate_random_pods()
        self.available_pods = self.inventory_pods.copy()
        self.delivered_pods = set()
        # Reset grid
        self.grid = np.zeros((self.grid_height, self.grid_width))
        for pod in self.inventory_pods:
            self.grid[pod] = 1
        self.select_new_target()
        return self.get_state()

    def get_state(self):
        """Get current state of the environment."""
        return {
            'grid': self.grid.copy(),
            'workstations': self.workstations.copy(),
            'inventory_pods': self.inventory_pods.copy(),
            'delivery_locations': self.delivery_locations.copy()
        }

    def step(self, actions):
        """Take a step in the environment for multiple agents.
        
        Args:
            actions: List of actions for each agent (0: up, 1: down, 2: left, 3: right)
            
        Returns:
            next_positions: List of new positions for each agent
            rewards: List of rewards for each agent
            dones: List of done flags for each agent
        """
        # Ensure visualization is initialized
        self._init_visualization()

        next_positions = []
        rewards = []
        dones = []
        
        movements = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        for i, action in enumerate(actions):
            # Get current position
            current_pos = self.workstations[i] if i < len(self.workstations) else self.robot_pos
            
            # Calculate new position
            dr, dc = movements[action]
            new_pos = (current_pos[0] + dr, current_pos[1] + dc)
            
            # Check if move is valid
            if (0 <= new_pos[0] < self.grid_height and 
                0 <= new_pos[1] < self.grid_width and 
                (self.grid[new_pos] == 0 or new_pos in self.available_pods)):
                next_positions.append(new_pos)
                reward = 0
                
                # Reward for getting closer to target pod
                if self.target_pod:
                    old_dist = abs(current_pos[0] - self.target_pod[0]) + abs(current_pos[1] - self.target_pod[1])
                    new_dist = abs(new_pos[0] - self.target_pod[0]) + abs(new_pos[1] - self.target_pod[1])
                    reward += 0.1 * (old_dist - new_dist)
                
                # Big reward for reaching target pod
                if new_pos == self.target_pod:
                    reward += 1.0
                    self.mark_pod_delivered()
                    self.select_new_target()
                
                rewards.append(reward)
            else:
                next_positions.append(current_pos)
                rewards.append(-0.1)  # Small penalty for invalid move
            
            # Check if episode is done
            dones.append(self.all_pods_delivered())
        
        # Update robot position for visualization
        if next_positions:
            self.robot_pos = next_positions[0]
        
        # Render the environment
        self.render()
        
        return next_positions, rewards, dones

    def close(self):
        """Close the environment."""
        if self.window:
            self.window.destroy()
            self.window = None
            self.canvas = None 