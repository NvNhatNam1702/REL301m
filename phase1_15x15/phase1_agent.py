import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch.nn.functional as F

class DQN(nn.Module):
    # Changed 'weight' to 'width' for clarity
    def __init__(self, input_channels, action_size, width, height):
        super(DQN, self).__init__()
        # CNN layers for flexible input size
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # A more robust way to compute the size after flattening
        flattened_size = self._get_conv_output(input_channels, width, height)

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    # Helper function to dynamically determine the size of the flattened layer
    def _get_conv_output(self, channels, width, height):
        with torch.no_grad():
            # Create a dummy tensor with the specified input shape
            input = torch.zeros(1, channels, height, width)
            # Pass it through the convolutional layers
            output = self.conv3(self.conv2(self.conv1(input)))
            # Return the total number of elements in the output tensor
            return output.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten the tensor for the fully connected layer
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

    
        channels, height, width = self.state_shape
        self.policy_net = DQN(channels, action_size, width, height).to(self.device)
        self.target_net = DQN(channels, action_size, width, height).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
    def preprocess_state(self, state):
        """Convert state to tensor without adding batch dimension for training."""
        return torch.FloatTensor(state).to(self.device)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def replay(self):
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack([self.preprocess_state(s) for s in states])
        next_states = torch.stack([self.preprocess_state(s) for s in next_states])
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filename):
        """Save model weights."""
        torch.save(self.policy_net.state_dict(), filename)
    
    def load(self, filename):
        """Load model weights."""
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())