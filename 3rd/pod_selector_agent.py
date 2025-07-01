import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
import itertools

class PodSelectorAgent(nn.Module):
    def __init__(self, state_shape, max_pods, lr=0.001, max_pod_distance=8):
        super().__init__()
        self.state_shape = state_shape
        self.max_pods = max_pods
        self.max_pod_distance = max_pod_distance  # Maximum Manhattan distance between pods in a batch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # CNN layers: assume state_shape = (20, 20) or (channels, 20, 20)
        if len(state_shape) == 2:
            in_channels = 1
            height, width = state_shape
        else:
            in_channels, height, width = state_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Calculate flattened size after conv layers
        dummy_input = torch.zeros(1, in_channels, height, width)
        with torch.no_grad():
            cnn_out_size = self.cnn(dummy_input).shape[1]
        self.fc1 = nn.Linear(cnn_out_size, 128)
        self.fc2 = nn.Linear(128, max_pods)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.memory = []
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.to(self.device)

    def _calculate_manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_batch(self, pod_indices, available_pods):
        """Check if a batch of pods is valid (pods are not too far apart)."""
        if len(pod_indices) != 3:
            return False
        
        # Get the actual pod positions
        batch_pods = [available_pods[i] for i in pod_indices]
        
        # Check pairwise distances
        for i in range(len(batch_pods)):
            for j in range(i + 1, len(batch_pods)):
                distance = self._calculate_manhattan_distance(batch_pods[i], batch_pods[j])
                if distance > self.max_pod_distance:
                    return False
        return True

    def _get_valid_batches(self, available_pods, scores):
        """Get all valid batches (pods not too far apart) sorted by score."""
        n = len(available_pods)
        if n < 3:
            return []
        
        valid_batches = []
        
        # Generate all possible combinations of 3 pods
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    batch_indices = [i, j, k]
                    if self._is_valid_batch(batch_indices, available_pods):
                        # Calculate average score for this batch
                        batch_score = (scores[i] + scores[j] + scores[k]) / 3.0
                        valid_batches.append((batch_score, batch_indices))
        
        # Sort by score (highest first)
        valid_batches.sort(key=lambda x: x[0], reverse=True)
        return valid_batches

    def forward(self, state):
        # state: (20, 20) or (channels, 20, 20)
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if len(state.shape) == 2:
            # (20, 20) -> (1, 1, 20, 20)
            state = state.unsqueeze(0).unsqueeze(0)
        elif len(state.shape) == 3:
            # (channels, 20, 20) -> (1, channels, 20, 20)
            state = state.unsqueeze(0)
        state = state.to(self.device)
        x = self.cnn(state)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits for each pod slot

    def _compute_total_path_cost(self, env, start_pos, batch_pods, delivery_point):
        """Compute the total path cost to collect all pods in batch_pods (in order) and deliver to delivery_point."""
        current = start_pos
        total_cost = 0
        for pod in batch_pods:
            path = env._plan_path(current, pod)
            if not path or len(path) < 2:
                return float('inf')  # unreachable
            total_cost += len(path) - 1
            current = pod
        # Path from last pod to delivery
        path = env._plan_path(current, delivery_point)
        if not path or len(path) < 2:
            return float('inf')
        total_cost += len(path) - 1
        return total_cost

    def act(self, state, available_pods, env, top_k=5):
        n = len(available_pods)
        if n < 3:
            return list(range(n))
        batch_candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    batch_indices = [i, j, k]
                    if self._is_valid_batch(batch_indices, available_pods):
                        batch_pods = [available_pods[idx] for idx in batch_indices]
                        cost = self._compute_total_path_cost(env, env.robot_pos, batch_pods, env.delivery_point)
                        if cost < float('inf'):
                            batch_candidates.append((cost, batch_indices))
        if not batch_candidates:
            # fallback: random batch
            return random.sample(range(n), 3)
        # Sort by cost (lowest first)
        batch_candidates.sort(key=lambda x: x[0])
        top_batches = batch_candidates[:top_k]
        # Use neural network to pick among top-k batches
        with torch.no_grad():
            logits = self.forward(state).cpu().numpy().flatten()
            best_score = -float('inf')
            best_batch = None
            for cost, batch_indices in top_batches:
                score = logits[batch_indices[0]] + logits[batch_indices[1]] + logits[batch_indices[2]]
                if score > best_score:
                    best_score = score
                    best_batch = batch_indices
            if best_batch is not None:
                return best_batch
        # fallback: lowest cost batch
        return top_batches[0][1]

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
            state_tensor = torch.FloatTensor(state).to(self.device)
            if len(state_tensor.shape) == 2:
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
            elif len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(0)
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
            'max_pods': self.max_pods,
            'max_pod_distance': self.max_pod_distance
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.max_pod_distance = checkpoint.get('max_pod_distance', 8)  # Default fallback 