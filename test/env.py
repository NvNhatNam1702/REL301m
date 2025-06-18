import numpy as np
import torch
import random
import time
import heapq
from typing import List, Tuple, Set
import tkinter as tk

class HybridPodDeliveryEnv:
    """Hybrid environment: DQN for pod collection, A* for delivery"""
    
    def __init__(self, seed=None, fixed_pod_positions=None):
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Grid dimensions
        self.grid_height = 9
        self.grid_width = 9
        
        # Define robot starting position (bottom left)
        self.robot_start_pos = (self.grid_height - 1, 0)
        self.robot_pos = self.robot_start_pos
        
        # Define delivery point (top right)
        self.delivery_point = (0, self.grid_width - 1)
        
        # Pod management
        self.num_total_pods = 15  # Total pods in environment
        self.pods_per_delivery = 3  # Pods to collect before delivery
        self.fixed_pod_positions = fixed_pod_positions  # Store fixed positions if provided
        self.pod_positions = self._generate_pods()
        self.collected_pods = set()
        self.delivered_pods = 0
        
        # Mode tracking
        self.mode = "COLLECTING"  # "COLLECTING" or "DELIVERING"
        self.current_path = []
        self.path_index = 0
        
        # Initialize visualization
        self.cell_size = 60
        self.window = None
        self.canvas = None
        self.render_enabled = True  # Flag to control rendering
        self.fast_render = False  # New flag for fast rendering
        self._init_visualization()
        
        # Training parameters
        self.max_steps = 500
        self.current_steps = 0
        self.current_reward = 0.0
        self.total_reward = 0.0
        
    def _init_visualization(self):
        """Initialize the visualization window and canvas."""
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("Hybrid Pod Collection & Delivery Environment")
            self.canvas = tk.Canvas(
                self.window,
                width=self.grid_width * self.cell_size,
                height=self.grid_height * self.cell_size,
                bg='white'
            )
            self.canvas.pack()
            
            # Add info label
            self.info_label = tk.Label(
                self.window,
                text="Mode: COLLECTING | Steps: 0 | Collected: 0/3 | Delivered: 0",
                font=("Arial", 12)
            )
            self.info_label.pack()

    def _generate_pods(self):
        """Generate pod positions, using fixed positions if provided."""
        if self.fixed_pod_positions is not None:
            return self.fixed_pod_positions.copy()
            
        possible_positions = [
            (r, c)
            for r in range(self.grid_height)
            for c in range(self.grid_width)
            if (r, c) != self.robot_start_pos and (r, c) != self.delivery_point
        ]
        return random.sample(possible_positions, self.num_total_pods)

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (within bounds and not blocked by uncollected pods)."""
        row, col = pos
        
        # Check bounds
        if not (0 <= row < self.grid_height and 0 <= col < self.grid_width):
            return False
        
        # During delivery mode, uncollected pods are obstacles
        if self.mode == "DELIVERING" and pos in self.pod_positions and pos not in self.collected_pods:
            return False
            
        return True

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions, avoiding uncollected pods during delivery."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # up, down, left, right
            new_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position(new_pos):
                neighbors.append(new_pos)
        return neighbors

    def _find_path_to_delivery(self) -> List[Tuple[int, int]]:
        """Find path from current position to delivery point using A*, avoiding uncollected pods."""
        start = self.robot_pos
        goal = self.delivery_point
        
        # Check if goal is reachable (not blocked by uncollected pod)
        if not self._is_valid_position(goal):
            print(f"Warning: Delivery point {goal} is blocked!")
            return []
        
        # Priority queue for A*
        frontier = []
        heapq.heappush(frontier, (0, start))
        
        # Keep track of where we came from
        came_from = {start: None}
        
        # Cost from start to current node
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
                
            for next_pos in self._get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self._heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        
        # Verify path is valid and complete
        if not path or path[0] != start:
            print(f"Warning: No valid path found from {start} to {goal}")
            uncollected = [p for p in self.pod_positions if p not in self.collected_pods]
            print(f"Uncollected pods (obstacles): {uncollected}")
            return []
        
        print(f"Found delivery path of length {len(path)}: {start} -> {goal}")
        return path

    def _find_alternative_delivery_path(self) -> List[Tuple[int, int]]:
        """
        Find an alternative path using weighted A* - pods have high cost but aren't impossible to traverse.
        This is a backup when normal pathfinding fails.
        """
        start = self.robot_pos
        goal = self.delivery_point
        
        print("Attempting alternative pathfinding with weighted costs...")
        
        # Try A* with higher cost for pod positions instead of complete blocking
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
                
            # Get all valid neighbors (including those with pods, but with higher cost)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (current[0] + dr, current[1] + dc)
                
                # Check bounds only
                if not (0 <= next_pos[0] < self.grid_height and 0 <= next_pos[1] < self.grid_width):
                    continue
                
                # Calculate cost - higher for pod positions
                new_cost = cost_so_far[current] + 1
                if next_pos in self.pod_positions and next_pos not in self.collected_pods:
                    new_cost += 100  # High cost for pod positions
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self._heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        
        if not path or path[0] != start:
            print("Alternative pathfinding also failed!")
            return []
        
        print(f"Found alternative delivery path of length {len(path)}")
        return path

    def render(self):
        """Render the current state of the environment."""
        if not self.render_enabled:
            return
            
        if self.fast_render:
            self.window.update()
            return
            
        self.canvas.delete("all")
        
        # Draw grid
        for i in range(self.grid_height + 1):
            self.canvas.create_line(
                0, i * self.cell_size,
                self.grid_width * self.cell_size, i * self.cell_size,
                fill="gray"
            )
        for i in range(self.grid_width + 1):
            self.canvas.create_line(
                i * self.cell_size, 0,
                i * self.cell_size, self.grid_height * self.cell_size,
                fill="gray"
            )

        # Draw pods (uncollected)
        for pod in self.pod_positions:
            if pod not in self.collected_pods:
                x1 = pod[1] * self.cell_size + 5
                y1 = pod[0] * self.cell_size + 5
                x2 = (pod[1] + 1) * self.cell_size - 5
                y2 = (pod[0] + 1) * self.cell_size - 5
                self.canvas.create_rectangle(x1, y1, x2, y2, fill='green')

        # Draw delivery point
        delivery_x = self.delivery_point[1] * self.cell_size + self.cell_size // 2
        delivery_y = self.delivery_point[0] * self.cell_size + self.cell_size // 2
        self.canvas.create_oval(
            delivery_x - self.cell_size // 3,
            delivery_y - self.cell_size // 3,
            delivery_x + self.cell_size // 3,
            delivery_y + self.cell_size // 3,
            fill='blue'
        )

        # Draw robot
        robot_x = self.robot_pos[1] * self.cell_size + self.cell_size // 2
        robot_y = self.robot_pos[0] * self.cell_size + self.cell_size // 2
        self.canvas.create_oval(
            robot_x - self.cell_size // 3,
            robot_y - self.cell_size // 3,
            robot_x + self.cell_size // 3,
            robot_y + self.cell_size // 3,
            fill='red'
        )

        # Draw path if in delivery mode
        if self.mode == "DELIVERING" and self.current_path:
            for i in range(len(self.current_path) - 1):
                start = self.current_path[i]
                end = self.current_path[i + 1]
                x1 = start[1] * self.cell_size + self.cell_size // 2
                y1 = start[0] * self.cell_size + self.cell_size // 2
                x2 = end[1] * self.cell_size + self.cell_size // 2
                y2 = end[0] * self.cell_size + self.cell_size // 2
                self.canvas.create_line(x1, y1, x2, y2, fill='gray', width=2)

        # Update info label
        self.info_label.config(
            text=f"Mode: {self.mode} | Steps: {self.current_steps} | "
                 f"Collected: {len(self.collected_pods)}/{len(self.pod_positions)} | "
                 f"Delivered: {self.delivered_pods} | "
                 f"Reward: {self.current_reward:.1f} | Total: {self.total_reward:.1f}"
        )

        self.window.update()

    def get_state(self):
        """Get the current state representation."""
        state = np.zeros((4, self.grid_height, self.grid_width), dtype=np.float32)
        
        # Channel 0: robot position
        state[0, self.robot_pos[0], self.robot_pos[1]] = 1.0
        
        # Channel 1: uncollected pod positions (only show pods that haven't been delivered)
        for pod in self.pod_positions:
            if pod not in self.collected_pods:
                state[1, pod[0], pod[1]] = 1.0
        
        # Channel 2: delivery point
        state[2, self.delivery_point[0], self.delivery_point[1]] = 1.0
        
        # Channel 3: mode indicator (1.0 for collecting, 0.0 for delivering)
        if self.mode == "COLLECTING":
            state[3, :, :] = 1.0
        
        return state

    def reset(self):
        """Reset the environment to its initial state."""
        self.robot_pos = self.robot_start_pos
        self.pod_positions = self._generate_pods()
        self.collected_pods = set()
        self.delivered_pods = 0
        self.mode = "COLLECTING"
        self.current_path = []
        self.path_index = 0
        self.current_steps = 0
        self.current_reward = 0.0
        self.total_reward = 0.0
        
        if self.render_enabled:
            self.render()
        
        return self.get_state()

    def step(self, action):
        """Take a step in the environment."""
        self.current_steps += 1
        reward = 0
        done = False
        
        if self.mode == "COLLECTING":
            reward, done = self._handle_collection_step(action)
        else:  # DELIVERING
            reward, done = self._handle_delivery_step()
        
        self.total_reward += reward
        
        if self.render_enabled:
            self.render()
        
        return self.get_state(), reward, done, {}

    def _handle_collection_step(self, action):
        """Handle a step in collection mode."""
        # Convert action to movement
        dr, dc = 0, 0
        if action == 0:  # Up
            dr = -1
        elif action == 1:  # Down
            dr = 1
        elif action == 2:  # Left
            dc = -1
        elif action == 3:  # Right
            dc = 1
        
        # Calculate new position
        new_pos = (self.robot_pos[0] + dr, self.robot_pos[1] + dc)
        
        # Check if new position is valid
        if not self._is_valid_position(new_pos):
            return -0.1, False  # Small penalty for invalid move
        
        # Update robot position
        self.robot_pos = new_pos
        
        # Check if pod is collected
        if self.robot_pos in self.pod_positions and self.robot_pos not in self.collected_pods:
            self.collected_pods.add(self.robot_pos)
            reward = 1.0  # Reward for collecting a pod
        else:
            reward = -0.01  # Small penalty for each step
        
        # Check if enough pods are collected for delivery
        if len(self.collected_pods) >= self.pods_per_delivery:
            self.mode = "DELIVERING"
            self.current_path = self._find_path_to_delivery()
            if not self.current_path:
                self.current_path = self._find_alternative_delivery_path()
            self.path_index = 0
        
        # Check if episode is done
        done = self._check_done()
        
        return reward, done

    def _handle_delivery_step(self):
        """Handle a step in delivery mode."""
        if not self.current_path:
            # If no path is found, try to find one
            self.current_path = self._find_path_to_delivery()
            if not self.current_path:
                self.current_path = self._find_alternative_delivery_path()
            self.path_index = 0
            
            if not self.current_path:
                return -1.0, True  # Penalty for failing to find a path
        
        # Move along the path
        if self.path_index < len(self.current_path):
            self.robot_pos = self.current_path[self.path_index]
            self.path_index += 1
            
            # Check if reached delivery point
            if self.robot_pos == self.delivery_point:
                # Remove delivered pods from the environment
                for pod in self.collected_pods:
                    self.pod_positions.remove(pod)
                self.delivered_pods += len(self.collected_pods)
                self.collected_pods.clear()
                self.mode = "COLLECTING"
                return 10.0, self._check_done()  # Large reward for successful delivery
        
        return -0.01, self._check_done()  # Small penalty for each step

    def _check_done(self):
        """Check if the episode is done."""
        return (self.current_steps >= self.max_steps or
                (self.mode == "COLLECTING" and len(self.pod_positions) == 0))

    def close(self):
        """Close the environment."""
        if self.window is not None:
            self.window.destroy()
            self.window = None
            self.canvas = None

    def get_pod_positions(self):
        """Get the current pod positions."""
        return self.pod_positions.copy()