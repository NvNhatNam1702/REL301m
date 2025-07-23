
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from collections import deque
import tkinter as tk

class WarehouseEnvMultiPod:
    def __init__(self, seed=None):
        self.seed(seed)
        self.grid_height = 10
        self.grid_width = 10
        self.max_pods_per_cell = 3
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        self.workstations = [(9, 0)]
        self.delivery_locations = [(0, 9)]
        self._generate_random_pods()
        self.cell_size = 40
        self.window = None
        self.canvas = None
        self._init_visualization()
        self.robot_pos = self.workstations[0]
        self.delivered_pods = 0
        self.path = []
        self.pods_carried = 0
        self.max_capacity = 3
        self.step_count = 0
        self.max_steps = 1500

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
        available_cells = [
            (r, c) for r in range(self.grid_height) for c in range(self.grid_width)
            if (r, c) not in self.workstations and (r, c) not in self.delivery_locations
        ]
        total_pods = 0
        for r, c in available_cells:
            if random.random() < 0.3:
                pods = random.randint(1, self.max_pods_per_cell)
                self.grid[r, c] = pods
                total_pods += pods
            else:
                self.grid[r, c] = 0
        remainder = total_pods % 3
        if remainder != 0:
            pods_to_add = 3 - remainder
            random.shuffle(available_cells)
            for r, c in available_cells:
                if pods_to_add == 0:
                    break
                if self.grid[r, c] < self.max_pods_per_cell:
                    addable = min(self.max_pods_per_cell - self.grid[r, c], pods_to_add)
                    self.grid[r, c] += addable
                    pods_to_add -= addable
            if pods_to_add > 0:
                for r, c in available_cells:
                    if pods_to_add == 0:
                        break
                    if self.grid[r, c] == 0:
                        addable = min(self.max_pods_per_cell, pods_to_add)
                        self.grid[r, c] = addable
                        pods_to_add -= addable

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
                    fill = "green"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill)
                if self.grid[r, c] > 0:
                    self.canvas.create_text(
                        (x1 + x2) // 2, (y1 + y2) // 2,
                        text=str(self.grid[r, c]), fill="black", font=("Arial", 14)
                    )
        robot_x = self.robot_pos[1] * self.cell_size + self.cell_size // 2
        robot_y = self.robot_pos[0] * self.cell_size + self.cell_size // 2
        self.canvas.create_oval(
            robot_x - self.cell_size//3,
            robot_y - self.cell_size//3,
            robot_x + self.cell_size//3,
            robot_y + self.cell_size//3,
            fill="red"
        )
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

    def reset(self):
        if self.window:
            self.window.destroy()
        self.window = None
        self.canvas = None
        self._init_visualization()
        self.robot_pos = self.workstations[0]
        self.path = []
        self._generate_random_pods()
        self.delivered_pods = 0
        self.pods_carried = 0
        self.step_count = 0
        return self.get_state_tensor()

    def get_state_tensor(self):
        pods = self.grid.astype(np.float32)
        robot = np.zeros_like(pods)
        robot[self.robot_pos[0], self.robot_pos[1]] = 1
        delivery = np.zeros_like(pods)
        for loc in self.delivery_locations:
            delivery[loc[0], loc[1]] = 1
        pods_carried = np.full_like(pods, self.pods_carried / self.max_capacity)
        state = np.stack([pods, robot, delivery, pods_carried], axis=0)
        return state

    def all_pods_delivered(self):
        return np.sum(self.grid) == 0

    def step(self, action, total_reward=None, render=False):
        self.step_count += 1
        movements = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = movements[action]
        new_pos = (self.robot_pos[0] + dr, self.robot_pos[1] + dc)
        reward = 0
        done = False
        if (0 <= new_pos[0] < self.grid_height and 0 <= new_pos[1] < self.grid_width):
            self.robot_pos = new_pos
            if self.grid[self.robot_pos] > 0 and self.pods_carried < self.max_capacity:
                pods_available = self.grid[self.robot_pos]
                capacity_left = self.max_capacity - self.pods_carried
                pods_to_pick = min(pods_available, capacity_left)
                self.grid[self.robot_pos] -= pods_to_pick
                self.pods_carried += pods_to_pick
                reward += 1 * pods_to_pick if np.sum(self.grid) > 3 else 2 * pods_to_pick  # Increased reward for sparse pods
            if self.robot_pos in self.delivery_locations and self.pods_carried > 0:
                reward += self.pods_carried
                self.delivered_pods += self.pods_carried
                self.pods_carried = 0
            else:
                reward = -0.1
        else:
            reward = -0.1
        done = self.all_pods_delivered() or self.step_count >= self.max_steps
        if render:
            self.render(total_reward=total_reward)
        return self.get_state_tensor(), reward, done

    def seed(self, seed=None):
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

class QNetwork(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, action_dim, lr=0.0001, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.1, epsilon_decay=0.995, buffer_size=50000, batch_size=64, 
                 target_update=100):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.q_net = QNetwork(state_shape[0], action_dim)
        self.target_net = QNetwork(state_shape[0], action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.steps = 0

    def select_action(self, state):
        if state[3, 0, 0] == 1.0:
            robot_r, robot_c = np.where(state[1] == 1)
            robot_r, robot_c = robot_r[0], robot_c[0]
            delivery_r, delivery_c = np.where(state[2] == 1)
            delivery_r, delivery_c = delivery_r[0], delivery_c[0]
            if robot_r > delivery_r:
                return 0
            elif robot_r < delivery_r:
                return 1
            elif robot_c < delivery_c:
                return 3
            elif robot_c > delivery_c:
                return 2
            else:
                return 0
        else:
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.stack(states))
        next_states = torch.FloatTensor(np.stack(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

if __name__ == "__main__":
    env = WarehouseEnvMultiPod(seed=42)
    agent = DQNAgent(state_shape=(4, 10, 10), action_dim=4)
    num_episodes = 1000
    render_frequency = 100

    recent_rewards = deque(maxlen=10)
    best_avg_reward = float('-inf')

    for episode in range(num_episodes):
        # Load best weights at the start of each episode if available
        if os.path.exists('best_model.pth'):
            agent.q_net.load_state_dict(torch.load('best_model.pth'))
            agent.target_net.load_state_dict(agent.q_net.state_dict())
        state = env.reset()
        total_reward = 0
        step = 0
        done = False

        while not done:
            action = agent.select_action(state)
            render = (episode % render_frequency == 0)
            next_state, reward, done = env.step(action, total_reward, render=render)
            agent.store_experience(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
            step += 1
            print(f"\rEpisode {episode+1:3d}, Step {step:4d}, Reward: {total_reward:6.2f}, Held Pods: {env.pods_carried:2d}", end='', flush=True)
        
        print()
        print(f"Episode {episode+1}, Total Reward: {total_reward}")
        agent.decay_epsilon()
        
        if episode % render_frequency == 0:
            env.render(total_reward=total_reward)
        
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.q_net.state_dict(), 'best_model.pth')
                print(f"Model saved with average reward: {avg_reward:.2f}")
