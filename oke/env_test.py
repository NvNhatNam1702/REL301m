import numpy as np
from warehouse_env_multipod import WarehouseEnvMultiPod as OriginalEnv
from test2 import WarehouseEnvMultiPod as Test2Env

# Set the seed for reproducibility
seed = 42

# Create an instance of the original environment (for GreedyBatchAStarAgent)
env_original = OriginalEnv(seed=seed)
grid_original = env_original.grid.copy()

# Create an instance of the test2 environment (for DQNAgent)
env_test2 = Test2Env(seed=seed)
grid_test2 = env_test2.grid.copy()

# Print the grids
print("Original Environment Grid (warehouse_env_multipod.py):")
print(grid_original)
print("\nTest2 Environment Grid (test2.py):")
print(grid_test2)

# Check if the grids are identical
are_equal = np.array_equal(grid_original, grid_test2)
print(f"\nAre the initial environment grids identical? {are_equal}")