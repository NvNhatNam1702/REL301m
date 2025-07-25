import numpy as np
from warehouse_env_multipod import WarehouseEnvMultiPod as OriginalEnv
# from test2 import WarehouseEnvMultiPod as Test2Env, DQNAgent
from greedy_batch_astar_agent import GreedyBatchAStarAgent
import torch
import random
from config import DQN_SHAPE, ENV_SHAPE
# Function to generate 100 unique random seeds
def generate_random_seeds(num_seeds=100, min_seed=1, max_seed=100000):
    random.seed(42)  # Fixed seed for reproducible seed generation
    seeds = random.sample(range(min_seed, max_seed + 1), num_seeds)
    return seeds

# Function to run GreedyBatchAStarAgent and return steps taken
def run_greedy_agent(env, seed):
    env.seed(seed)
    env.reset()
    agent = GreedyBatchAStarAgent(env)
    total_steps = 0
    while not env.all_pods_delivered():
        start = env.robot_pos
        batch = agent.select_greedy_batch(start)
        if not batch:
            break
        plan = agent.plan_batch_path(start, batch)
        if plan is None:
            break
        path, _ = plan
        for pos in path:
            env.update_robot_position(pos)
            if pos in batch and env.grid[pos] > 0:
                env.mark_pod_delivered(pos)
            total_steps += 1
        env.robot_pos = agent.delivery
    return total_steps

# Function to run DQNAgent and return steps and total reward
def run_dqn_agent(env, seed, model_path='best_model.pth'):
    env.seed(seed)
    state = env.reset()
    agent = DQNAgent(
        state_shape=DQN_SHAPE, 
        action_dim=4,
        epsilon_start=0.1,  # Set epsilon to 0.1
        epsilon_end=0.1,    # Ensure epsilon stays at 0.1
        epsilon_decay=1.0   # No decay
    )
    agent.q_net.load_state_dict(torch.load(model_path))
    total_reward = 0
    steps = 0
    done = False
    while not done and steps < env.max_steps:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action, total_reward=total_reward, render=False)
        state = next_state
        total_reward += reward
        steps += 1
    return steps, total_reward

# Main comparison script
if __name__ == "__main__":
    # Generate 100 random seeds
    seeds = generate_random_seeds(num_seeds=100)

    greedy_steps_list = []
    dqn_steps_list = []
    dqn_reward_list = []
    valid_seeds = []

    for seed in seeds:
        print(f"\nTesting with seed: {seed}")

        # Run GreedyBatchAStarAgent
        env_original = OriginalEnv(seed=seed)
        greedy_steps = run_greedy_agent(env_original, seed)
        greedy_steps_list.append(greedy_steps)
        print(f"GreedyBatchAStarAgent steps: {greedy_steps}")

        # Run DQNAgent
        env_test2 = Test2Env(seed=seed)
        dqn_steps, dqn_reward = run_dqn_agent(env_test2, seed)
        dqn_steps_list.append(dqn_steps)
        dqn_reward_list.append(dqn_reward)
        print(f"DQNAgent steps: {dqn_steps}, total reward: {dqn_reward}")

        # Track valid seeds where DQNAgent steps < 1500
        if dqn_steps < 1500:
            valid_seeds.append(seed)

    # Filter results for valid seeds
    valid_greedy_steps = [greedy_steps_list[i] for i, seed in enumerate(seeds) if seed in valid_seeds]
    valid_dqn_steps = [dqn_steps_list[i] for i, seed in enumerate(seeds) if seed in valid_seeds]
    valid_dqn_rewards = [dqn_reward_list[i] for i, seed in enumerate(seeds) if seed in valid_seeds]

    # Compute and display average metrics for valid seeds
    if valid_seeds:
        avg_greedy_steps = np.mean(valid_greedy_steps)
        avg_dqn_steps = np.mean(valid_dqn_steps)
        avg_dqn_reward = np.mean(valid_dqn_rewards)
        print(f"\nNumber of valid seeds (DQNAgent steps < 1500): {len(valid_seeds)}")
        print("\nAverage Metrics (over valid seeds):")
        print(f"GreedyBatchAStarAgent average steps: {avg_greedy_steps:.2f}")
        print(f"DQNAgent average steps: {avg_dqn_steps:.2f}")
        print(f"DQNAgent average reward: {avg_dqn_reward:.2f}")
    else:
        print("\nNo valid seeds: DQNAgent reached max steps (1500) for all seeds")
