import numpy as np
import tkinter as tk
import random

class WarehouseEnvMultiPod:
    def __init__(self, seed=None):
        self.seed(seed)
        self.grid_height = 10
        self.grid_width = 10

        # Each cell can have 0 to 3 pods
        self.max_pods_per_cell = 3
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)  # stores pod count

        # Define workstations and delivery locations
        self.workstations = [
            (9, 0),  # Bottom left
        ]
        self.delivery_locations = [(0, 9)]  # Top right

        # Randomly generate pods in the grid
        self._generate_random_pods()

        # Visualization
        self.cell_size = 40
        self.window = None
        self.canvas = None
        self._init_visualization()

        # Robot state
        self.robot_pos = self.workstations[0]
        self.target_pod = self._select_new_target()
        self.delivered_pods = 0  # total pods delivered
        self.path = []
        self.pods_carried = 0
        self.max_capacity = 3

    def _init_visualization(self):
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("Warehouse Environment - MultiPod")
            self.canvas = tk.Canvas(
                self.window,
                width=self.grid_width * self.cell_size,
                height=self.grid_height * self.cell_size
            )
            self.canvas.pack()

    def _generate_random_pods(self):
        # Initial pod generation
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if (r, c) in self.workstations or (r, c) in self.delivery_locations:
                    self.grid[r, c] = 0
                else:
                    # 30% chance to have pods, otherwise 0
                    if random.random() < 0.3:
                        self.grid[r, c] = random.randint(1, self.max_pods_per_cell)
                    else:
                        self.grid[r, c] = 0
        
        # Adjust total pods to be a multiple of 3
        total_pods = np.sum(self.grid)
        remainder = total_pods % 3
        if remainder != 0:
            pods_to_add = 3 - remainder
            # List of cells excluding workstations and delivery locations
            available_cells = [(r, c) for r in range(self.grid_height) for c in range(self.grid_width)
                              if (r, c) not in self.workstations and (r, c) not in self.delivery_locations]
            random.shuffle(available_cells)
            # Add pods to cells with capacity
            for r, c in available_cells:
                if pods_to_add == 0:
                    break
                if self.grid[r, c] < self.max_pods_per_cell:
                    addable = min(self.max_pods_per_cell - self.grid[r, c], pods_to_add)
                    self.grid[r, c] += addable
                    pods_to_add -= addable

    def _select_new_target(self):
        # Find all cells with at least 1 pod
        pod_cells = [(r, c) for r in range(self.grid_height) for c in range(self.grid_width)
                     if self.grid[r, c] > 0]
        if not pod_cells:
            return None
        return random.choice(pod_cells)

    def mark_pod_delivered(self, pos):
        # Remove one pod from the cell
        if self.grid[pos] > 0:
            self.grid[pos] -= 1
            self.delivered_pods += 1

    def all_pods_delivered(self):
        return np.sum(self.grid) == 0

    def render(self, total_reward=None):
        self.canvas.delete("all")
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                fill = "white"
                if (r, c) in self.workstations:
                    fill = "blue"
                elif (r, c) in self.delivery_locations:
                    fill = "yellow"
                elif self.grid[r, c] > 0:
                    if (r, c) == self.target_pod:
                        fill = "purple"
                    else:
                        fill = "green"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill)
                # Draw pod count if any
                if self.grid[r, c] > 0:
                    self.canvas.create_text(
                        (x1 + x2) // 2, (y1 + y2) // 2,
                        text=str(self.grid[r, c]), fill="black", font=("Arial", 14)
                    )
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
        self.robot_pos = new_pos

    def clear_path(self):
        self.path = []

    def reset(self):
        if self.window:
            self.window.destroy()
        self.window = None
        self.canvas = None
        self._init_visualization()
        self.robot_pos = self.workstations[0]
        self.path = []
        self._generate_random_pods()
        self.target_pod = self._select_new_target()
        self.delivered_pods = 0
        self.pods_carried = 0
        return self.get_state()

    def get_state(self):
        return {
            'grid': self.grid.copy(),
            'workstations': self.workstations.copy(),
            'delivery_locations': self.delivery_locations.copy(),
            'robot_pos': self.robot_pos,
            'delivered_pods': self.delivered_pods,
            'pods_carried': self.pods_carried
        }

    def step(self, action, total_reward=None):
        # action: 0=up, 1=down, 2=left, 3=right
        movements = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }
        dr, dc = movements[action]
        new_pos = (self.robot_pos[0] + dr, self.robot_pos[1] + dc)
        reward = 0
        done = False
        # Check bounds
        if (0 <= new_pos[0] < self.grid_height and 0 <= new_pos[1] < self.grid_width):
            self.robot_pos = new_pos
            # If on a pod cell and not full, pick up one pod
            if self.grid[self.robot_pos] > 0 and self.pods_carried < self.max_capacity:
                self.grid[self.robot_pos] -= 1
                self.pods_carried += 1
                reward += 0.1  # small reward for picking up a pod
            # If on delivery location and carrying pods, deliver all
            if self.robot_pos in self.delivery_locations and self.pods_carried > 0:
                reward += self.pods_carried  # reward for delivery
                self.delivered_pods += self.pods_carried
                self.pods_carried = 0
            # Optionally: small reward for moving closer to any pod or delivery (not using target_pod)
        else:
            reward = -0.1  # Penalty for invalid move
        done = self.all_pods_delivered() and self.pods_carried == 0
        self.render(total_reward=total_reward)
        return self.get_state(), reward, done

    def close(self):
        if self.window:
            self.window.destroy()
            self.window = None
            self.canvas = None

    def seed(self, seed=None):
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
            except ImportError:
                pass