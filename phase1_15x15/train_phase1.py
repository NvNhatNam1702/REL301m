import numpy as np
# Assuming the environment can be initialized with a specific size
from phase1_env import Phase1Env 
from phase1_agent import PodCollectionAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

def save_training_info(episode_rewards, episode_lengths, pod_collection_rates, save_dir, best_metrics):
    """Save training metrics and information."""
    training_info = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'pod_collection_rates': pod_collection_rates,
        'best_metrics': best_metrics,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Save metrics
    with open(os.path.join(save_dir, 'training_metrics.json'), 'w') as f:
        json.dump(training_info, f)
    
    # Plot and save metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.axhline(y=best_metrics['reward'], color='r', linestyle='--', 
                label=f'Best Reward: {best_metrics["reward"]:.1f}')
    plt.legend()
    
    plt.subplot(132)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(133)
    plt.plot(pod_collection_rates)
    plt.title('Pod Collection Rate')
    plt.xlabel('Episode')
    plt.ylabel('Collection Rate')
    plt.axhline(y=best_metrics['collection_rate'], color='r', linestyle='--',
                label=f'Best Rate: {best_metrics["collection_rate"]:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

# --- MODIFICATION 1: Add height and width parameters ---
def train(episodes=1000, height=9, width=9, render_interval=100, eval_interval=5, window_size=10):
    # Create save directory with size information
    save_dir = f"saved_models_{height}x{width}"
    os.makedirs(save_dir, exist_ok=True)
    
    # --- MODIFICATION 2: Initialize environment with the given size ---
    # This assumes your Phase1Env constructor can accept height and width
    env = Phase1Env(height=height, width=width)
    
    # --- MODIFICATION 3: Define state_shape and action_size dynamically ---
    # The number of channels (3) is assumed to be constant
    state_shape = (3, env.height, env.width) 
    action_size = 4  # up, down, left, right (assuming this is constant)
    
    agent = PodCollectionAgent(state_shape, action_size)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    pod_collection_rates = []
    
    # Best model tracking
    best_metrics = {
        'collection_rate': 0.0,
        'reward': float('-inf'),
        'episode': 0
    }
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    # Training loop
    for episode in tqdm(range(episodes)):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if episode % render_interval == 0:
                env.render()
            
            if done:
                break
        
        if episode % 10 == 0:
            agent.update_target_network()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Handle division by zero if there are no pods
        num_pods = len(env.pod_positions)
        pod_collection_rate = len(env.collected_pods) / num_pods if num_pods > 0 else 0
        pod_collection_rates.append(pod_collection_rate)
        
        if (episode + 1) % eval_interval == 0 and episode >= window_size:
            recent_rewards = episode_rewards[-window_size:]
            recent_collection_rates = pod_collection_rates[-window_size:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_collection_rate = sum(recent_collection_rates) / len(recent_collection_rates)
            
            if (avg_collection_rate > best_metrics['collection_rate'] or 
                (avg_collection_rate == best_metrics['collection_rate'] and 
                 avg_reward > best_metrics['reward'])):
                
                best_metrics = {
                    'collection_rate': avg_collection_rate,
                    'reward': avg_reward,
                    'episode': episode
                }
                agent.save(best_model_path)
                print(f"\nNew best model saved for size {height}x{width}!")
                print(f"Average Collection Rate: {avg_collection_rate:.2f}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Episode: {episode}")
                
                save_training_info(
                    episode_rewards,
                    episode_lengths,
                    pod_collection_rates,
                    save_dir,
                    best_metrics
                )
        
        if episode % 5 == 0:
            print(f"\nEpisode {episode}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Best Collection Rate for {height}x{width}: {best_metrics['collection_rate']:.2f}")

    env.close()
    return agent

# --- MODIFICATION 4: Update the main execution block ---
if __name__ == "__main__":
    # Train the model with the default 9x9 size
    print("--- Starting training for 9x9 grid ---")
    train(episodes=1000, height=15, width=15)
