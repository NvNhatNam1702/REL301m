import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_channels, action_size, grid_size=9):
        super(DQN, self).__init__()
        # CNN layers for input
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Compute the size after flattening
        self.flattened_size = 64 * grid_size * grid_size
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PodCollectionAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape  # (channels, height, width)
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_shape[0], action_size, self.state_shape[1]).to(self.device)
        self.target_net = DQN(self.state_shape[0], action_size, self.state_shape[1]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
    def preprocess_state(self, state, add_batch_dim=False):
        """Convert state to tensor with optional batch dimension."""
        state_tensor = torch.FloatTensor(state).to(self.device)
        if add_batch_dim:
            state_tensor = state_tensor.unsqueeze(0)
        return state_tensor
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = self.preprocess_state(state, add_batch_dim=True)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def replay(self):
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack([self.preprocess_state(s) for s in states])
        next_states = torch.stack([self.preprocess_state(s) for s in next_states])
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filename):
        """Save model weights and metadata."""
        save_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_shape': self.state_shape,
            'action_size': self.action_size
        }
        torch.save(save_dict, filename)
    
    def load(self, filename):
        """Load model weights and metadata."""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            
            # Handle both old and new format
            if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
                # New format with metadata
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                
                # Verify state shape and action size match
                if checkpoint.get('state_shape') != self.state_shape:
                    raise ValueError(f"State shape mismatch: loaded {checkpoint['state_shape']}, expected {self.state_shape}")
                if checkpoint.get('action_size') != self.action_size:
                    raise ValueError(f"Action size mismatch: loaded {checkpoint['action_size']}, expected {self.action_size}")
            else:
                # Old format (just policy network state dict)
                print("Loading model in old format (policy network only)")
                self.policy_net.load_state_dict(checkpoint)
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            print(f"Successfully loaded model from {filename}")
            print(f"Current epsilon: {self.epsilon:.3f}")
            
        except Exception as e:
            print(f"Error loading model from {filename}: {str(e)}")
            raise 