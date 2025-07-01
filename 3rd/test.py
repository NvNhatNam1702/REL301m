import numpy as np
import torch
import os
import time
import argparse
from env import HybridPodDeliveryEnv
from pod_selector_agent import PodSelectorAgent

def test(model_path=None, num_episodes=5, render_delay=0.01):
    """Test the agent with a saved model (pod selection version)."""
    env = HybridPodDeliveryEnv()
    state_shape = (4, env.grid_height, env.grid_width)
    max_pods = env.num_total_pods
    agent = PodSelectorAgent(state_shape, max_pods, max_pod_distance=8)
    if model_path is None:
        model_path = os.path.join("saved_models", "best_model.pth")
    # print("Testing Pod Selection System")
    # print("=" * 30)
    try:
        agent.load(model_path)
        agent.epsilon = 0.0
        env.render_enabled = True
        env.fast_render = False
        total_rewards = []
        total_steps = []
        total_collected = []
        total_delivered = []
        total_test_steps = 0
        episode_step_counts = []
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            episode_steps = 0
            while not done:
                available_pods = env.get_uncollected_pods()
                if len(available_pods) < 3 or env.mode != "COLLECTING":
                    _, reward, done, info = env.step(0)
                    total_reward += reward
                    step_incr = info.get('steps', 1) if isinstance(info, dict) else 1
                    steps += step_incr
                    total_test_steps += step_incr
                    episode_steps += step_incr
                    continue
                batch_action = agent.act(state, available_pods, env)
                next_state, reward, done, info = env.step_select_pod_batch(batch_action)
                state = next_state
                total_reward += reward
                step_incr = info.get('steps', 1) if isinstance(info, dict) else 1
                steps += step_incr
                total_test_steps += step_incr
                episode_steps += step_incr
                env.render()
                time.sleep(render_delay)
            total_rewards.append(total_reward)
            total_steps.append(steps)
            total_collected.append(len(env.collected_pods))
            total_delivered.append(env.delivered_pods)
            episode_step_counts.append(episode_steps)
            # print(f"\nEpisode {episode + 1}")
            # print(f"Total Reward: {total_reward:.2f}")
            # print(f"Steps: {steps}")
            # print(f"Pods Collected: {len(env.collected_pods)}")
            # print(f"Pods Delivered: {env.delivered_pods}")
            # print(f"Steps this episode: {episode_steps}")
        # print("\nAverage Performance:")
        # print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}")
        # print(f"Average Steps: {sum(total_steps)/len(total_steps):.2f}")
        # print(f"Average Pods Collected: {sum(total_collected)/len(total_collected):.2f}")
        # print(f"Average Pods Delivered: {sum(total_delivered)/len(total_delivered):.2f}")
        # print(f"Total test steps: {total_test_steps}")
        # print(f"Average steps per episode: {sum(episode_step_counts)/len(episode_step_counts):.2f}")
        print(f"Final rewards per episode: {total_rewards}")
        print(f"Total steps per episode: {total_steps}")
        print(f"Steps per episode: {episode_step_counts}")
    except Exception as e:
        # print(f"Error loading model from {model_path}: {str(e)}")
        # print("Error: Could not load model from", model_path)
        # print("Please train the agent first or provide a valid model path.")
        return
    env.window.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the pod selection system')
    parser.add_argument('--model_path', type=str, help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test')
    parser.add_argument('--render_delay', type=float, default=0.01, help='Delay between rendering steps')
    args = parser.parse_args()
    test(args.model_path, args.episodes, args.render_delay) 