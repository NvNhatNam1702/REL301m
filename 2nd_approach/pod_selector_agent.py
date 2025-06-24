import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F

class PodSelectorAgent(nn.Module):
    def __init__(self, state_shape, max_pods, lr=0.001):
        super().__init__()
        self.state_shape = state_shape
        self.max_pods = max_pods
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(np.prod(state_shape), 128)
        self.fc2 = nn.Linear(128, max_pods)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.memory = []
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.to(self.device)

    def forward(self, state):
        x = torch.FloatTensor(state).to(self.device).flatten().unsqueeze(0)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits for each pod slot

    def act(self, state, available_pods):
        n = len(available_pods)
        if n < 3:
            return list(range(n))
        if random.random() < self.epsilon:
            return random.sample(range(n), 3)
        with torch.no_grad():
            logits = self.forward(state).cpu().numpy().flatten()
            scores = logits[:n]
            top3 = np.argsort(scores)[-3:][::-1]
            return top3.tolist()

    def remember(self, state, action, reward, next_state, done, available_pods, next_available_pods):
        self.memory.append((state, action, reward, next_state, done, available_pods, next_available_pods))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done, available_pods, next_available_pods in batch:
            n = len(available_pods)
            if n < 3:
                continue
            state_tensor = torch.FloatTensor(state).to(self.device).flatten().unsqueeze(0)
            q_values = self.forward(state)[0, :n]
            # Q-value for the batch: mean of the 3 selected pods
            q_selected = q_values[action].mean()
            with torch.no_grad():
                next_n = len(next_available_pods)
                if next_n >= 3:
                    next_q_values = self.forward(next_state)[0, :next_n]
                    next_top3 = torch.topk(next_q_values, 3).values.mean().item()
                else:
                    next_top3 = 0.0
                target = reward + (0 if done else self.gamma * next_top3)
            loss = F.smooth_l1_loss(q_selected, torch.tensor(target).to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_shape': self.state_shape,
            'max_pods': self.max_pods
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon) 