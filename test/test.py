import numpy as np
import torch
import os
import time
import argparse
from env import HybridPodDeliveryEnv
from phase1_agent import PodCollectionAgent

def test(model_path=None, num_episodes=5, render_delay=0.01):
    """Test the agent with a saved model."""
    # Initialize environment and agent
    env = HybridPodDeliveryEnv()
    state_shape = (4, env.grid_height, env.grid_width)
    action_size = 4
    agent = PodCollectionAgent(state_shape, action_size)
    
    # If no model path provided, use the best model from saved_models
    if model_path is None:
        model_path = os.path.join("saved_models", "best_model.pth")
    
    print("Testing Hybrid System")
    print("=" * 30)
    
    try:
        # Load the model
        agent.load(model_path)
        agent.epsilon = 0.0  # No exploration during testing
        
        # Enable rendering
        env.render_enabled = True
        env.fast_render = False
        
        total_rewards = []
        total_steps = []
        total_collected = []
        total_delivered = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
                env.render()
                time.sleep(render_delay)
            
            total_rewards.append(total_reward)
            total_steps.append(steps)
            total_collected.append(len(env.collected_pods))
            total_delivered.append(env.delivered_pods)
            
            print(f"\nEpisode {episode + 1}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Steps: {steps}")
            print(f"Pods Collected: {len(env.collected_pods)}")
            print(f"Pods Delivered: {env.delivered_pods}")
        
        # Print average performance
        print("\nAverage Performance:")
        print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}")
        print(f"Average Steps: {sum(total_steps)/len(total_steps):.2f}")
        print(f"Average Pods Collected: {sum(total_collected)/len(total_collected):.2f}")
        print(f"Average Pods Delivered: {sum(total_delivered)/len(total_delivered):.2f}")
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        print("Error: Could not load model from", model_path)
        print("Please train the agent first or provide a valid model path.")
        return
    
    # Keep window open after testing
    env.window.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the hybrid pod collection and delivery system')
    parser.add_argument('--model_path', type=str, help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test')
    parser.add_argument('--render_delay', type=float, default=0.01, help='Delay between rendering steps')
    
    args = parser.parse_args()
    test(args.model_path, args.episodes, args.render_delay) 