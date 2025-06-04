import numpy as np
from phase1_env import Phase1Env
from phase1_agent import PodCollectionAgent
import torch

def test_model(model_path, num_episodes=5):
    """Test a saved model."""
    # Initialize environment and agent
    env = Phase1Env()
    # The state is now a 3-channel grid (3, 9, 9)
    state_shape = (3, 9, 9)
    action_size = 4
    agent = PodCollectionAgent(state_shape, action_size)
    
    # Load model
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during testing
    
    print(f"\nTesting model: {model_path}")
    print("=" * 50)
    
    total_rewards = []
    total_steps = []
    total_pods = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
        total_pods.append(len(env.collected_pods))
        
        print(f"\nEpisode {episode + 1}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps: {steps}")
        print(f"Pods Collected: {len(env.collected_pods)}/{len(env.pod_positions)}")
    
    # Print average performance
    print("\nAverage Performance:")
    print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Average Steps: {sum(total_steps)/len(total_steps):.2f}")
    print(f"Average Pods Collected: {sum(total_pods)/len(total_pods):.2f}")
    
    env.close()

if __name__ == "__main__":
    # Test the best model
    test_model("saved_models/best_model.pth") 