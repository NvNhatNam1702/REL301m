import numpy as np
import tkinter as tk
from typing import List, Tuple, Set
import random



class Phase1Env:
    """Environment for Phase 1: Learning to collect pods efficiently"""
    def __init__(self, grid_height, grid_width):
        # Grid dimensions
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        # Define robot starting position (bottom left)
        self.robot_pos = (self.grid_height - 1, 0)
        
        # Define pod positions (randomly placed)
        self.num_pods = 10
        self.pod_positions = self._generate_pods()
        
        # Initialize visualization
        self.cell_size = 60  # Larger cells for better visibility
        self.window = None
        self.canvas = None
        self._init_visualization()
        
        # Track collected pods
        self.collected_pods = set()
        
        # Training parameters
        self.max_steps = 200
        self.current_steps = 0
        self.current_reward = 0.0  # Current step reward
        self.total_reward = 0.0    # Total episode reward

    def _init_visualization(self):
        """Initialize the visualization window and canvas."""
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("Phase 1: Pod Collection Environment")
            self.canvas = tk.Canvas(
                self.window,
                width=self.grid_width * self.cell_size,
                height=self.grid_height * self.cell_size,
                bg='white'
            )
            self.canvas.pack()
            
            # Add episode info label
            self.info_label = tk.Label(
                self.window,
                text="Steps: 0 | Pods: 0/10 | Current Reward: 0.0 | Total Reward: 0.0",
                font=("Arial", 12)
            )
            self.info_label.pack()

    def _generate_pods(self):
        """Generate random pod positions, avoiding the robot's starting position."""
        possible_positions = [
            (r, c)
            for r in range(self.grid_height)
            for c in range(self.grid_width)
            if (r, c) != self.robot_pos
        ]
        return random.sample(possible_positions, self.num_pods)

    def render(self):
        """Render the grid environment."""
        self.canvas.delete("all")

        # Draw grid lines
        for i in range(self.grid_height + 1):
            self.canvas.create_line(
                0, i * self.cell_size,
                self.grid_width * self.cell_size, i * self.cell_size,
                fill='gray'
            )
        for i in range(self.grid_width + 1):
            self.canvas.create_line(
                i * self.cell_size, 0,
                i * self.cell_size, self.grid_height * self.cell_size,
                fill='gray'
            )

        # Draw pods (uncollected)
        for pod in self.pod_positions:
            if pod not in self.collected_pods:
                x1 = pod[1] * self.cell_size + 5
                y1 = pod[0] * self.cell_size + 5
                x2 = (pod[1] + 1) * self.cell_size - 5
                y2 = (pod[0] + 1) * self.cell_size - 5
                self.canvas.create_rectangle(x1, y1, x2, y2, fill='green')

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

        # Update info label
        self.info_label.config(
            text=f"Steps: {self.current_steps} | Pods: {len(self.collected_pods)}/{len(self.pod_positions)} | "
                 f"Current Reward: {self.current_reward:.1f} | Total Reward: {self.total_reward:.1f}"
        )

        self.window.update()

    def get_state(self):
        state = np.zeros((3, self.grid_height, self.grid_width), dtype=np.float32)
        # Channel 0: robot position
        state[0, self.robot_pos[0], self.robot_pos[1]] = 1.0
        # Channel 1: uncollected pod positions
        for pod in self.pod_positions:
            if pod not in self.collected_pods:
                state[1, pod[0], pod[1]] = 1.0
        # Channel 2: collected pod positions (optional, for memory)
        for pod in self.collected_pods:
            state[2, pod[0], pod[1]] = 1.0
        return state

    def reset(self):
        """Reset the environment to initial state."""
        # Reset robot position
        self.robot_pos = (self.grid_height - 1, 0)
        # Generate new pod positions
        self.pod_positions = self._generate_pods()
        # Reset collected pods
        self.collected_pods = set()
        # Reset steps and rewards
        self.current_steps = 0
        self.current_reward = 0.0
        self.total_reward = 0.0
        # Update visualization
        self.render()
        return self.get_state()

    def step(self, action):
        """Take a step in the environment.
        
        Args:
            action: 0 (up), 1 (down), 2 (left), 3 (right)
            
        Returns:
            next_state: New state after action
            reward: Reward for the action
            done: Whether episode is done
        """
        self.current_steps += 1
        
        # Define movements
        movements = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        # Get new position
        dr, dc = movements[action]
        new_pos = (self.robot_pos[0] + dr, self.robot_pos[1] + dc)
        
        # Initialize reward
        reward = -0.1  # Small penalty for each step
        
        # Check if move is valid
        if (0 <= new_pos[0] < self.grid_height and 
            0 <= new_pos[1] < self.grid_width):
            # Remove grid manipulation
            self.robot_pos = new_pos
        else:
            reward = -5.0  # Penalty for hitting boundaries (Increased)
        
        # Check for pod collection
        if self.robot_pos in self.pod_positions and self.robot_pos not in self.collected_pods:
            self.collected_pods.add(self.robot_pos)
            reward = 10.0  # Reward for collecting a pod
        
        # Check if done
        done = False
        if len(self.collected_pods) == len(self.pod_positions):
            reward += 50.0  # Bonus for collecting all pods
            done = True
        elif self.current_steps >= self.max_steps:
            done = True
        
        # Update rewards
        self.current_reward = reward
        self.total_reward += reward
        
        # Update visualization
        self.render()
        
        return self.get_state(), reward, done

    def close(self):
        """Close the environment."""
        if self.window:
            self.window.destroy()
            self.window = None
            self.canvas = None

if __name__ == "__main__":
    # Test the environment
    env = Phase1Env()
    
    # Run a few random steps
    for _ in range(20):
        action = random.randint(0, 3)
        state, reward, done = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        print(f"Collected pods: {len(env.collected_pods)}/{len(env.pod_positions)}")
        print("---")
        
        if done:
            break
    
    # Keep window open
    env.window.mainloop() 