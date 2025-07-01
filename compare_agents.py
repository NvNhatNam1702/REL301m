import random
import sys
import os
import itertools
sys.path.append('./3rd')
sys.path.append('./now')

from env import HybridPodDeliveryEnv
from pod_selector_agent import PodSelectorAgent
from warehouse_env_astar import WarehouseEnvAStar
from agent_astar import AStarAgent

NUM_SEEDS = 1000
MODEL_PATH = './3rd/saved_models/best_model.pth'

def total_path_length(order, start, delivery, agent):
    """Calculate total path length for a given order of pods."""
    current = start
    total = 0
    for pod in order:
        agent.env.target_pod = pod
        path = agent.find_path(current, pod)
        if not path:
            return float('inf')  # unreachable
        total += len(path) - 1
        current = pod
    # After last pod, go to delivery
    agent.env.target_pod = None
    path = agent.find_path(current, delivery)
    if not path:
        return float('inf')
    total += len(path) - 1
    return total

def find_optimal_batch_and_order(remaining_pods, start, delivery, agent, batch_size):
    """Find the optimal batch and order of pods to collect."""
    best_batch = None
    best_order = None
    best_cost = float('inf')
    
    # Limit the number of pods to consider for performance
    max_pods_to_consider = min(8, len(remaining_pods))  # Only consider closest 8 pods
    if len(remaining_pods) > max_pods_to_consider:
        # Sort pods by distance to start position and take closest ones
        pod_distances = [(pod, abs(pod[0] - start[0]) + abs(pod[1] - start[1])) for pod in remaining_pods]
        pod_distances.sort(key=lambda x: x[1])
        remaining_pods = [pod for pod, _ in pod_distances[:max_pods_to_consider]]
    
    # Try all batch sizes from batch_size down to 1 (if fewer pods remain)
    for k in range(min(batch_size, len(remaining_pods)), 0, -1):
        # Limit combinations for performance
        max_combinations = 50  # Limit to prevent excessive computation
        combinations_tried = 0
        
        for batch in itertools.combinations(remaining_pods, k):
            if combinations_tried >= max_combinations:
                break
            combinations_tried += 1
            
            # Limit permutations for performance
            max_permutations = 20  # Limit to prevent excessive computation
            permutations_tried = 0
            
            for order in itertools.permutations(batch):
                if permutations_tried >= max_permutations:
                    break
                permutations_tried += 1
                
                cost = total_path_length(order, start, delivery, agent)
                if cost < best_cost:
                    best_cost = cost
                    best_batch = batch
                    best_order = order
                    
                    # Early termination if we find a good enough solution
                    if best_cost < 50:  # If path is reasonably short, accept it
                        return best_batch, best_order, best_cost
        
        if best_batch is not None:
            break  # Prefer largest possible batch
    
    return best_batch, best_order, best_cost

def run_3rd_agent(seed, agent):
    """Run the trained neural agent from 3rd folder."""
    env = HybridPodDeliveryEnv(seed=seed)
    env.render_enabled = False  # Disable rendering for speed
    state = env.reset()
    done = False
    total_steps = 0
    
    while not done and total_steps < env.max_steps:
        available_pods = env.get_uncollected_pods()
        if len(available_pods) < 3 or env.mode != "COLLECTING":
            _, _, done, info = env.step(0)
            step_incr = info.get('steps', 1) if isinstance(info, dict) else 1
            total_steps += step_incr
        else:
            batch_action = agent.act(state, available_pods, env)
            _, _, done, info = env.step_select_pod_batch(batch_action)
            step_incr = info.get('steps', 1) if isinstance(info, dict) else 1
            total_steps += step_incr
    
    env.close()
    return total_steps

def run_now_metaheuristic(seed):
    """Run the metaheuristic agent from now folder."""
    env = WarehouseEnvAStar(seed=seed)
    agent = AStarAgent(env)
    batch_size = 3
    start = env.robot_pos
    delivery = env.delivery_locations[0]
    total_steps = 0

    while env.inventory_pods:
        # Find the optimal batch and order
        batch, order, cost = find_optimal_batch_and_order(env.inventory_pods, start, delivery, agent, batch_size)
        if not batch:
            break  # No reachable pods left
        
        total_steps += cost

        current = start
        for pod in order:
            # Move to pod
            segment = agent.find_path(current, pod)
            if segment:
                current = pod
            # Remove pod from inventory
            if pod in env.inventory_pods:
                env.inventory_pods.remove(pod)
                env.grid[pod] = 0
        
        # Move to delivery
        segment = agent.find_path(current, delivery)
        if segment:
            start = delivery  # Next batch starts from delivery location
    
    env.close()
    return total_steps

def main():
    """Main comparison function."""
    print("Loading trained model from 3rd folder...")
    
    # Load the trained agent for 3rd
    env = HybridPodDeliveryEnv()
    state_shape = (4, env.grid_height, env.grid_width)
    max_pods = env.num_total_pods
    agent = PodSelectorAgent(state_shape, max_pods, max_pod_distance=8)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train the agent first or provide a valid model path.")
        return
    
    try:
        agent.load(MODEL_PATH)
        agent.epsilon = 0.0  # Use trained policy only
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"Generating {NUM_SEEDS} random seeds...")
    seeds = [random.randint(0, 1_000_000) for _ in range(NUM_SEEDS)]
    
    # Run all 3rd agent seeds first
    print("Running 3rd agent on all seeds...")
    third_results = {}
    for idx, seed in enumerate(seeds):
        try:
            steps = run_3rd_agent(seed, agent)
            third_results[seed] = steps
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{NUM_SEEDS} seeds for 3rd agent...")
        except Exception as e:
            print(f"Error processing seed {seed} for 3rd agent: {e}")
            third_results[seed] = 0
    
    # Run all now agent seeds
    print("Running now agent on all seeds...")
    now_results = {}
    for idx, seed in enumerate(seeds):
        try:
            steps = run_now_metaheuristic(seed)
            now_results[seed] = steps
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{NUM_SEEDS} seeds for now agent...")
        except Exception as e:
            print(f"Error processing seed {seed} for now agent: {e}")
            now_results[seed] = 0
    
    # Compare results
    print("Comparing results...")
    matches = 0
    mismatches = 0
    total_3rd_steps = 0
    total_now_steps = 0
    valid_seeds = 0
    
    for seed in seeds:
        steps_3rd = third_results[seed]
        steps_now = now_results[seed]
        
        # Skip seeds where either agent failed (returned 0 steps)
        if steps_3rd == 0 or steps_now == 0:
            print(f"Skipping seed {seed}: 3rd={steps_3rd} steps, now={steps_now} steps (failure detected)")
            continue
        
        valid_seeds += 1
        total_3rd_steps += steps_3rd
        total_now_steps += steps_now
        
        if steps_3rd == steps_now:
            matches += 1
        else:
            mismatches += 1
            print(f"Seed {seed}: 3rd={steps_3rd} steps, now={steps_now} steps")
    
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Total seeds processed: {NUM_SEEDS}")
    print(f"Valid seeds (no failures): {valid_seeds}")
    print(f"Failed seeds: {NUM_SEEDS - valid_seeds}")
    print(f"Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    print(f"Match rate: {matches/valid_seeds*100:.2f}% (of valid seeds)")
    print(f"\nAverage steps - 3rd agent: {total_3rd_steps/valid_seeds:.2f}")
    print(f"Average steps - now agent: {total_now_steps/valid_seeds:.2f}")
    print(f"Performance difference: {abs(total_3rd_steps - total_now_steps)/valid_seeds:.2f} steps")

if __name__ == "__main__":
    main() 