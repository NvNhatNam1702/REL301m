# --- Environment Constants ---
GRID_HEIGHT = 6
GRID_WIDTH = 6
# BUG FIX: The original workstations/delivery locations were outside the 6x6 grid.
# I've adjusted them to be within the grid.
WORKSTATIONS = [(5, 0)]
DELIVERY_LOCATIONS = [(0, 5)]
MAX_CAPACITY = 3
MAX_PODS_PER_CELL = 3
POD_GENERATION_PROB = 0.3
MAX_STEPS_PER_EPISODE = 1500

# --- Reward/Penalty Constants ---
REWARD_PICKUP_NORMAL = 1
REWARD_PICKUP_SPARSE = 2
SPARSE_POD_THRESHOLD = 3 # Threshold below which sparse reward is given
REWARD_DELIVERY_MULTIPLIER = 1.0 # Multiplied by number of pods delivered
PENALTY_MOVE = -0.1

# --- DQN Agent Hyperparameters ---
LEARNING_RATE = 0.0001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 100

# --- Neural Network Architecture ---
# Input shape: (num_channels, height, width) -> (4, GRID_HEIGHT, GRID_WIDTH)
DQN_SHAPE = (4, GRID_HEIGHT, GRID_WIDTH)
ACTION_DIM = 4
CONV1_OUT_CHANNELS = 32
CONV2_OUT_CHANNELS = 64
FC1_OUT_FEATURES = 128

# --- Training Constants ---
NUM_EPISODES = 1000
SEED = 42
RENDER_FREQUENCY = 100
REWARD_AVERAGE_WINDOW = 10
MODEL_PATH = 'best_model.pth'

# --- Visualization Constants ---
CELL_SIZE = 40
