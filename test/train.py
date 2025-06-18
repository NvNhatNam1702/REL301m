import numpy as np
import torch
import time
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from env import HybridPodDeliveryEnv
from phase1_agent import PodCollectionAgent
from tqdm import tqdm

def save_training_info(episode_rewards, episode_lengths, pod_collection_rates, delivery_rates, save_dir, best_metrics):
    """Save training metrics and information."""
    training_info = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'pod_collection_rates': pod_collection_rates,
        'delivery_rates': delivery_rates,
        'best_metrics': best_metrics,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Save metrics
    with open(os.path.join(save_dir, 'training_metrics.json'), 'w') as f:
        json.dump(training_info, f)
    
    # Plot and save metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.axhline(y=best_metrics['reward'], color='r', linestyle='--', 
                label=f'Best Reward: {best_metrics["reward"]:.1f}')
    plt.legend()
    
    plt.subplot(222)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(223)
    plt.plot(pod_collection_rates)
    plt.title('Pod Collection Rate')
    plt.xlabel('Episode')
    plt.ylabel('Collection Rate')
    plt.axhline(y=best_metrics['collection_rate'], color='r', linestyle='--',
                label=f'Best Rate: {best_metrics["collection_rate"]:.2f}')
    plt.legend()
    
    plt.subplot(224)
    plt.plot(delivery_rates)
    plt.title('Pod Delivery Rate')
    plt.xlabel('Episode')
    plt.ylabel('Delivery Rate')
    plt.axhline(y=best_metrics['delivery_rate'], color='r', linestyle='--',
                label=f'Best Rate: {best_metrics["delivery_rate"]:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def train(episodes=1000, render_interval=50, eval_interval=5, window_size=10, render_delay=0.01):
    """Train the agent for a specified number of episodes."""
    # Create save directory
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = HybridPodDeliveryEnv()
    state_shape = (4, env.grid_height, env.grid_width)  # 4 channels: robot, pods, delivery point, mode
    action_size = 4  # Up, down, left, right
    agent = PodCollectionAgent(state_shape, action_size)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    pod_collection_rates = []
    delivery_rates = []
    
    # Best model tracking
    best_metrics = {
        'collection_rate': 0.0,
        'delivery_rate': 0.0,
        'reward': float('-inf'),
        'episode': 0
    }
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    print("Starting Hybrid Pod Collection & Delivery Training")
    print("=" * 60)
    
    # Enable rendering during training
    env.render_enabled = True
    env.fast_render = False  # Disable fast render to see the visualization
    
    for episode in tqdm(range(episodes)):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        # Determine if this episode should be rendered
        should_render = episode % render_interval == 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            # Only store experiences and learn during collection phase
            if env.mode == "COLLECTING":
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                
                # Update target network during collection phase
                agent.update_target_network()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render every step if this is a rendered episode
            if should_render:
                env.render()
                time.sleep(render_delay)
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Calculate rates safely
        total_pods = len(env.pod_positions)
        if total_pods > 0:
            pod_collection_rate = len(env.collected_pods) / total_pods
            delivery_rate = env.delivered_pods / total_pods
        else:
            # If all pods are delivered, rates are 1.0
            pod_collection_rate = 1.0
            delivery_rate = 1.0
            
        pod_collection_rates.append(pod_collection_rate)
        delivery_rates.append(delivery_rate)
        
        # Evaluate and save best model
        if (episode + 1) % eval_interval == 0:
            # Calculate average metrics over recent episodes
            recent_rewards = episode_rewards[-window_size:]
            recent_collection_rates = pod_collection_rates[-window_size:]
            recent_delivery_rates = delivery_rates[-window_size:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_collection_rate = sum(recent_collection_rates) / len(recent_collection_rates)
            avg_delivery_rate = sum(recent_delivery_rates) / len(recent_delivery_rates)
            
            # Save if this is the best model so far
            # First check pod collection rate, then delivery rate, then reward if equal
            if (avg_collection_rate > best_metrics['collection_rate'] or 
                (avg_collection_rate == best_metrics['collection_rate'] and 
                 avg_delivery_rate > best_metrics['delivery_rate']) or
                (avg_collection_rate == best_metrics['collection_rate'] and 
                 avg_delivery_rate == best_metrics['delivery_rate'] and 
                 avg_reward > best_metrics['reward'])):
                
                best_metrics = {
                    'collection_rate': avg_collection_rate,
                    'delivery_rate': avg_delivery_rate,
                    'reward': avg_reward,
                    'episode': episode
                }
                agent.save(best_model_path)
                print(f"\nNew best model saved!")
                print(f"Average Collection Rate: {avg_collection_rate:.2f}")
                print(f"Average Delivery Rate: {avg_delivery_rate:.2f}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Episode: {episode}")
                
                # Save training info
                save_training_info(
                    episode_rewards,
                    episode_lengths,
                    pod_collection_rates,
                    delivery_rates,
                    save_dir,
                    best_metrics
                )
        
        # Print progress
        if episode % 5 == 0:
            print(f"\nEpisode {episode}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Steps: {steps}")
            print(f"Pods Collected: {len(env.collected_pods)}/{len(env.pod_positions)}")
            print(f"Pods Delivered: {env.delivered_pods}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Best Collection Rate: {best_metrics['collection_rate']:.2f}")
            print(f"Best Delivery Rate: {best_metrics['delivery_rate']:.2f}")
            print(f"Best Reward: {best_metrics['reward']:.2f}")
    
    print("\nTraining completed!")
    # Keep window open after training
    env.window.mainloop()

if __name__ == "__main__":
    train() 