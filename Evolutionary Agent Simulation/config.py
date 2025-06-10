# config.py

class SimulationConfig:
    # Environment
    PLANE_WIDTH = 800
    PLANE_HEIGHT = 600
    BOUNDARY_DAMAGE = 10
    FOOD_SPAWN_RATE = 0.3
    INITIAL_FOOD_COUNT = 20
    FOOD_HP_GAIN = 60
    FOOD_RADIUS = 5
    FOOD_LIFETIME = 200

    # Agent
    INITIAL_HP = 200
    HP_DECAY_RATE = 0.6  # Increased for a greater challenge
    EAT_HP_COST = 10
    MAX_SIGHT_DISTANCE = 100
    SIGHT_ANGLE = 90  # Degrees
    NUM_SIGHT_RAYS = 9 # Odd number for a central ray
    BASE_MOVEMENT_SPEED = 1.0
    MAX_MOVEMENT_SPEED = 5.0
    BASE_ROTATION_SPEED = 2.0  # Degrees per tick
    MAX_ROTATION_SPEED = 10.0 # Degrees per tick
    SPEED_HP_COST_FACTOR = 0.1
    ATTACK_HP_COST = 40
    ATTACK_RANGE = 20
    ATTACK_DAMAGE = 80
    DEATH_FOOD_DROP_AMOUNT = 5
    DEATH_FOOD_DROP_SCATTER_RADIUS = 30
    AGENT_RADIUS = 10

    # Entity Color Values (for sight input)
    COLOR_FOOD = [1.0, 0.0, 0.0]  # Red
    COLOR_AGENT = [0.0, 1.0, 0.0] # Green
    COLOR_BOUNDARY = [0.0, 0.0, 1.0] # Blue
    COLOR_EMPTY = [0.0, 0.0, 0.0] # Black

    # Reinforcement Learning
    RNN_HIDDEN_SIZE = 64
    LEARNING_RATE = 0.001
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95
    PPO_EPSILON = 0.2
    PPO_EPOCHS = 10
    MINIBATCH_SIZE = 64
    N_STEPS = 4096 # Increased for more stable rollouts
    ENTROPY_COEFFICIENT = 0.01

    # Simulation
    MAX_TICKS_PER_EPISODE = 2000
    NUM_AGENTS = 5
    NUM_EPISODES = 1000

    # Visualization
    RENDER_FPS = 60
    RENDER_TRAINING_INTERVAL = 10

    # Device for PyTorch
    DEVICE = 'cuda' # 'cuda' for GPU, 'cpu' for CPU