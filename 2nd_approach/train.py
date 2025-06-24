import numpy as np
import torch
import time
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from env import HybridPodDeliveryEnv
from pod_selector_agent import PodSelectorAgent
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

def train(episodes=2000, render_interval=0, eval_interval=10, window_size=20, render_delay=0.01):
    """Train the agent for a specified number of episodes (pod selection version)."""
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    env = HybridPodDeliveryEnv()
    state_shape = (4, env.grid_height, env.grid_width)
    max_pods = env.num_total_pods
    agent = PodSelectorAgent(state_shape, max_pods)
    episode_rewards = []
    episode_lengths = []
    pod_collection_rates = []
    delivery_rates = []
    episode_step_counts = []
    best_metrics = {
        'collection_rate': 0.0,
        'delivery_rate': 0.0,
        'reward': float('-inf'),
        'episode': 0
    }
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    # print("Starting Pod Selection Training (Large Environment)")
    # print("=" * 80)
    # print(f"Grid Size: {env.grid_height}x{env.grid_width}")
    # print(f"Total Pods: {env.num_total_pods}")
    # print(f"Pods per Delivery: {env.pods_per_delivery}")
    # print(f"Max Steps per Episode: {env.max_steps}")
    # print("=" * 80)
    env.render_enabled = False
    env.fast_render = True
    total_train_steps = 0
    for episode in tqdm(range(episodes)):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        episode_steps = 0
        should_render = render_interval > 0 and episode % render_interval == 0
        if should_render:
            env.render_enabled = True
            env.fast_render = False
        else:
            env.render_enabled = False
            env.fast_render = True
        while not done:
            available_pods = env.get_uncollected_pods()
            if len(available_pods) < 3 or env.mode != "COLLECTING":
                _, reward, done, info = env.step(0)  # dummy action for delivery
                total_reward += reward
                step_incr = info.get('steps', 1) if isinstance(info, dict) else 1
                steps += step_incr
                total_train_steps += step_incr
                episode_steps += step_incr
                continue
            batch_action = agent.act(state, available_pods)
            next_state, reward, done, info = env.step_select_pod_batch(batch_action)
            next_available_pods = env.get_uncollected_pods()
            agent.remember(state, batch_action, reward, next_state, done, available_pods, next_available_pods)
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            state = next_state
            total_reward += reward
            step_incr = info.get('steps', 1) if isinstance(info, dict) else 1
            steps += step_incr
            total_train_steps += step_incr
            episode_steps += step_incr
            if should_render:
                env.render()
                time.sleep(render_delay)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_step_counts.append(episode_steps)
        print(f"Episode {episode+1}: Reward = {total_reward}, Steps = {steps}")
        total_pods = env.num_total_pods
        if total_pods > 0:
            pod_collection_rate = len(env.collected_pods) / total_pods
            delivery_rate = env.delivered_pods / total_pods
        else:
            pod_collection_rate = 1.0
            delivery_rate = 1.0
        pod_collection_rates.append(pod_collection_rate)
        delivery_rates.append(delivery_rate)
        if (episode + 1) % eval_interval == 0:
            recent_rewards = episode_rewards[-window_size:]
            recent_collection_rates = pod_collection_rates[-window_size:]
            recent_delivery_rates = delivery_rates[-window_size:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_collection_rate = sum(recent_collection_rates) / len(recent_collection_rates)
            avg_delivery_rate = sum(recent_delivery_rates) / len(recent_delivery_rates)
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
                # print(f"\nNew best model saved!")
                # print(f"Average Collection Rate: {avg_collection_rate:.2f}")
                # print(f"Average Delivery Rate: {avg_delivery_rate:.2f}")
                # print(f"Average Reward: {avg_reward:.2f}")
                # print(f"Episode: {episode}")
                save_training_info(
                    episode_rewards,
                    episode_lengths,
                    pod_collection_rates,
                    delivery_rates,
                    save_dir,
                    {**best_metrics, 'total_train_steps': total_train_steps, 'episode_step_counts': episode_step_counts}
                )
        if episode % 10 == 0:
            # print(f"\nEpisode {episode}")
            # print(f"Total Reward: {total_reward:.2f}")
            # print(f"Steps: {steps}")
            # print(f"Pods Collected: {len(env.collected_pods)}/{len(env.pod_positions)}")
            # print(f"Pods Delivered: {env.delivered_pods}")
            # print(f"Epsilon: {agent.epsilon:.3f}")
            # print(f"Best Collection Rate: {best_metrics['collection_rate']:.2f}")
            # print(f"Best Delivery Rate: {best_metrics['delivery_rate']:.2f}")
            # print(f"Best Reward: {best_metrics['reward']:.2f}")
            pass
    # print("\nTraining completed!")
    # print(f"Total training steps: {total_train_steps}")
    # print(f"Average steps per episode: {sum(episode_step_counts)/len(episode_step_counts):.2f}")
    # print(f"Steps per episode: {episode_step_counts}")
    print(f"Final rewards per episode: {episode_rewards}")
    print(f"Total steps per episode: {episode_lengths}")
    print(f"Steps per episode: {episode_step_counts}")
    env.window.mainloop()

if __name__ == "__main__":
    train(render_interval=1) 