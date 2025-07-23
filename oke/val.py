import torch
import numpy as np
from test2 import WarehouseEnvMultiPod
from test2 import DQNAgent

# Define evaluation seeds for consistency
eval_seeds = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

# Initialize the agent and load the trained model
agent = DQNAgent(state_shape=(4, 10, 10), action_dim=4)
try:
    agent.q_net.load_state_dict(torch.load('best_model.pth'))
    agent.q_net.eval()  # Set the network to evaluation mode
    print("Loaded best_model.pth successfully")
except FileNotFoundError:
    print("Error: best_model.pth not found")
    exit(1)

# Set epsilon to 0 for greedy policy during evaluation
agent.epsilon = 0.1

# Metrics storage
rewards = []
successes = []
steps_list = []

# Evaluate over multiple seeds
for seed in eval_seeds:
    env = WarehouseEnvMultiPod(seed=seed)
    state = env.reset()
    total_reward = 0
    step = 0
    done = False

    print(f"\nEvaluation Seed {seed}: Initial State Shape = {state.shape}")

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        step += 1

        # Debug output every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: Action = {action}, Reward = {reward:.2f}, Total Reward = {total_reward:.2f}")

    steps_taken = env.step_count
    success = env.all_pods_delivered()
    rewards.append(total_reward)
    successes.append(success)
    steps_list.append(steps_taken)

    print(f"Seed {seed}: Total Reward = {total_reward:.2f}, Steps = {steps_taken}, Success = {success}")

# Calculate and print results
average_reward = np.mean(rewards)
std_reward = np.std(rewards)
success_rate = np.mean(successes)
average_steps = np.mean(steps_list)

print("\nEvaluation Summary:")
print(f"Average Total Reward: {average_reward:.2f} ± {std_reward:.2f}")
print(f"Success Rate: {success_rate:.2f}")
print(f"Average Steps: {average_steps:.2f}")