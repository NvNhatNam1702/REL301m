import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from warehouse_env_astar import WarehouseEnvAStar
import time

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    """Deep Q-Network for the warehouse environment"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class WarehouseRLEnv(WarehouseEnvAStar):
    """Enhanced warehouse environment for RL training"""
    def __init__(self):
        super().__init__()
        self.max_steps = 1000
        self.current_steps = 0
        self.collected_pods = set()
        self.carrying_pods = []
        self.max_carry_capacity = 3
        self.last_delivery_step = 0
        
    def get_state_vector(self):
        """Convert environment state to feature vector for RL agent"""
        state = []
        
        # Robot position (normalized)
        state.extend([self.robot_pos[0] / self.grid_height, self.robot_pos[1] / self.grid_width])
        
        # Target pod position (if exists)
        if self.target_pod:
            state.extend([self.target_pod[0] / self.grid_height, self.target_pod[1] / self.grid_width])
        else:
            state.extend([0, 0])
        
        # Delivery location
        delivery_pos = self.delivery_locations[0]
        state.extend([delivery_pos[0] / self.grid_height, delivery_pos[1] / self.grid_width])
        
        # Distance to target pod
        if self.target_pod:
            dist = abs(self.robot_pos[0] - self.target_pod[0]) + abs(self.robot_pos[1] - self.target_pod[1])
            state.append(dist / (self.grid_height + self.grid_width))
        else:
            state.append(0)
        
        # Distance to delivery
        delivery_dist = abs(self.robot_pos[0] - delivery_pos[0]) + abs(self.robot_pos[1] - delivery_pos[1])
        state.append(delivery_dist / (self.grid_height + self.grid_width))
        
        # Number of pods carried (normalized)
        state.append(len(self.carrying_pods) / self.max_carry_capacity)
        
        # Number of pods remaining (normalized)
        remaining_pods = len(self.inventory_pods) - len(self.collected_pods)
        state.append(remaining_pods / self.num_pods)
        
        # Steps since last delivery (normalized)
        steps_since_delivery = self.current_steps - self.last_delivery_step
        state.append(min(steps_since_delivery, 100) / 100)
        
        # Surrounding cell features (3x3 grid around robot)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = self.robot_pos[0] + dr, self.robot_pos[1] + dc
                if 0 <= r < self.grid_height and 0 <= c < self.grid_width:
                    if self.grid[r, c] == 1:  # Pod
                        state.append(1)
                    elif (r, c) in self.delivery_locations:  # Delivery
                        state.append(0.5)
                    else:  # Empty
                        state.append(0)
                else:  # Out of bounds
                    state.append(-1)
        
        return np.array(state, dtype=np.float32)
    
    def reset(self):
        """Reset environment for new episode"""
        super().reset()
        self.current_steps = 0
        self.collected_pods = set()
        self.carrying_pods = []
        self.last_delivery_step = 0
        return self.get_state_vector()
    
    def step_rl(self, action):
        """Take a step in the environment for RL training"""
        self.current_steps += 1
        
        # Define movements
        movements = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (0, 0)    # stay (for pickup/delivery actions)
        }
        
        old_pos = self.robot_pos
        reward = -0.01  # Small time penalty
        done = False
        
        # Execute movement
        if action in movements:
            dr, dc = movements[action]
            new_pos = (old_pos[0] + dr, old_pos[1] + dc)
            
            # Check if move is valid
            if (0 <= new_pos[0] < self.grid_height and 
                0 <= new_pos[1] < self.grid_width and 
                (self.grid[new_pos] == 0 or new_pos in self.inventory_pods or 
                 new_pos in self.delivery_locations)):
                self.robot_pos = new_pos
                
                # Reward for getting closer to objective
                if self.carrying_pods:
                    # Moving towards delivery when carrying pods
                    delivery_pos = self.delivery_locations[0]
                    old_dist = abs(old_pos[0] - delivery_pos[0]) + abs(old_pos[1] - delivery_pos[1])
                    new_dist = abs(new_pos[0] - delivery_pos[0]) + abs(new_pos[1] - delivery_pos[1])
                    reward += 0.1 * (old_dist - new_dist)
                elif self.target_pod:
                    # Moving towards target pod when not carrying
                    old_dist = abs(old_pos[0] - self.target_pod[0]) + abs(old_pos[1] - self.target_pod[1])
                    new_dist = abs(new_pos[0] - self.target_pod[0]) + abs(new_pos[1] - self.target_pod[1])
                    reward += 0.1 * (old_dist - new_dist)
            else:
                reward -= 0.1  # Penalty for invalid move
        
        # Check for pod pickup
        if (self.robot_pos in self.inventory_pods and 
            self.robot_pos not in self.collected_pods and
            len(self.carrying_pods) < self.max_carry_capacity):
            self.carrying_pods.append(self.robot_pos)
            self.collected_pods.add(self.robot_pos)
            self.grid[self.robot_pos] = 0  # Remove pod from grid
            reward += 1.0  # Reward for picking up pod
            
            # Select new target if this was the target
            if self.robot_pos == self.target_pod:
                self.select_new_target()
        
        # Check for delivery
        if (self.robot_pos in self.delivery_locations and self.carrying_pods):
            delivered_count = len(self.carrying_pods)
            self.carrying_pods = []
            self.last_delivery_step = self.current_steps
            reward += 2.0 * delivered_count  # Reward for delivery
            
            # Bonus for efficient delivery (multiple pods)
            if delivered_count > 1:
                reward += 1.0 * (delivered_count - 1)
        
        # Check if all pods collected and delivered
        if (len(self.collected_pods) == len(self.inventory_pods) and 
            not self.carrying_pods):
            reward += 10.0  # Big bonus for completing task
            done = True
        
        # Check for timeout
        if self.current_steps >= self.max_steps:
            done = True
            reward -= 5.0  # Penalty for timeout
        
        # Update visualization occasionally
        if self.current_steps % 50 == 0:
            self.render()
        
        return self.get_state_vector(), reward, done

class DQNAgent:
    """DQN Agent for warehouse navigation"""
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(100000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.batch_size = 64
        self.gamma = 0.95
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=1000):
    """Train the DQN agent"""
    env = WarehouseRLEnv()
    state_size = len(env.get_state_vector())
    action_size = 5  # up, down, left, right, stay
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    scores_window = deque(maxlen=100)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step_rl(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores_window.append(total_reward)
        scores.append(total_reward)
        
        # Train the agent
        agent.replay()
        
        # Update target network periodically
        if episode % 100 == 0:
            agent.update_target_network()
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores_window)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores

def test_agent(agent, episodes=5):
    """Test the trained agent"""
    env = WarehouseRLEnv()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nTesting Episode {episode + 1}")
        
        while True:
            # Use greedy policy (no exploration)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, done = env.step_rl(action)
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render more frequently during testing
            if steps % 10 == 0:
                env.render()
                time.sleep(0.1)  # Slow down for visualization
            
            if done:
                break
        
        print(f"Episode {episode + 1} completed in {steps} steps with total reward: {total_reward:.2f}")
        print(f"Pods collected: {len(env.collected_pods)}/{len(env.inventory_pods)}")
        
        # Keep window open for viewing
        time.sleep(2)
    
    env.window.mainloop()

def main():
    """Main training and testing loop"""
    print("Starting RL Warehouse Training...")
    
    # Train the agent
    agent, scores = train_agent(episodes=2000)
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    # Moving average
    window_size = 100
    moving_avg = [np.mean(scores[max(0, i-window_size):i+1]) for i in range(len(scores))]
    plt.plot(moving_avg)
    plt.title('Moving Average Score')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    
    plt.tight_layout()
    plt.show()
    
    print("Training completed! Testing the agent...")
    
    # Test the trained agent
    test_agent(agent, episodes=3)

if __name__ == "__main__":
    main()