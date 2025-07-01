import argparse
from env import HybridPodDeliveryEnv

def main():
    parser = argparse.ArgumentParser(description='Run HybridPodDeliveryEnv with a specific seed.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    env = HybridPodDeliveryEnv(seed=args.seed)
    print(f"Seed: {args.seed}")
    print(f"Pod positions: {env.pod_positions}")

if __name__ == '__main__':
    main() 