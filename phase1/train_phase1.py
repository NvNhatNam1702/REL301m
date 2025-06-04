import numpy as np
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

def train(episodes=1000, render_interval=100, eval_interval=5, window_size=10):
    # Create save directory
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = Phase1Env()
    
    # State shape for CNN: (channels, height, width)
    state_shape = (3, 9, 9)
    action_size = 4  # up, down, left, right
    
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
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.replay()
            
            # Update metrics
            total_reward += reward
            steps += 1
            
            # Update state
            state = next_state
            
            # Render every render_interval episodes
            if episode % render_interval == 0:
                env.render()
            
            if done:
                break
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        pod_collection_rate = len(env.collected_pods) / len(env.pod_positions)
        pod_collection_rates.append(pod_collection_rate)
        
        # Evaluate and save best model
        if (episode + 1) % eval_interval == 0:
            # Calculate average metrics over recent episodes
            recent_rewards = episode_rewards[-window_size:]
            recent_collection_rates = pod_collection_rates[-window_size:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_collection_rate = sum(recent_collection_rates) / len(recent_collection_rates)
            
            # Save if this is the best model so far
            # First check pod collection rate, then reward if equal
            if (avg_collection_rate > best_metrics['collection_rate'] or 
                (avg_collection_rate == best_metrics['collection_rate'] and 
                 avg_reward > best_metrics['reward'])):
                
                best_metrics = {
                    'collection_rate': avg_collection_rate,
                    'reward': avg_reward,
                    'episode': episode
                }
                agent.save(best_model_path)
                print(f"\nNew best model saved!")
                print(f"Average Collection Rate: {avg_collection_rate:.2f}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Episode: {episode}")
                
                # Save training info
                save_training_info(
                    episode_rewards,
                    episode_lengths,
                    pod_collection_rates,
                    save_dir,
                    best_metrics
                )
        
        # Print progress
        if episode % 5 == 0:
            print(f"\nEpisode {episode}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Steps: {steps}")
            print(f"Pods Collected: {len(env.collected_pods)}/{len(env.pod_positions)}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Best Collection Rate: {best_metrics['collection_rate']:.2f}")
            print(f"Best Reward: {best_metrics['reward']:.2f}")
    
    env.close()
    return agent

if __name__ == "__main__":
    # Train the model
    train() 