import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from controller_neural import *
from scipy.optimize import differential_evolution

# --- EvoGym Setup ---
NUM_GENERATIONS = 20  # Number of generations to evolve
STEPS = 500
SEED = 42  # Set random seed for reproducibility
np.random.seed(SEED)
random.seed(SEED)

SCENARIO = 'DownStepper-v0'
#SCENARIO = 'ObstacleTraverser-v0'

robot_structure = np.array([ 
[1,3,1,0,0],
[4,1,3,2,2],
[3,4,4,4,4],
[3,0,0,3,2],
[0,0,0,0,2]
])

connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]  # Observation size
output_size = env.action_space.shape[0]  # Action size

# ---- CONTROLLER ----
brain = NeuralController(input_size, output_size)

# ---- WEIGHT UTILITIES ----
def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])

def reshape_weights(flat_vector, model):
    shapes = [p.shape for p in model.parameters()]
    new_weights = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        new_weights.append(flat_vector[idx:idx+size].reshape(shape))
        idx += size
    return new_weights

# ---- FITNESS FUNCTION ----
def evaluate_fitness(weights, view=False):
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    viewer = EvoViewer(env.sim)
    viewer.track_objects('robot')
    state = env.reset()[0]
    t_reward = 0
    for t in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        if view:
            viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        t_reward += reward
        if terminated or truncated:
            break
    viewer.close()
    env.close()
    return t_reward

# ---- DE FITNESS WRAPPER ----
def de_fitness(flat_weights):
    weights = reshape_weights(flat_weights, brain)
    return -evaluate_fitness(weights)

# ---- DE SETUP ----
initial_weights = get_weights(brain)
flat_len = len(flatten_weights(initial_weights))
bounds = [(-2, 2)] * flat_len

result = differential_evolution(
    de_fitness,
    bounds,
    strategy='best1bin',
    maxiter=NUM_GENERATIONS,
    popsize=15,
    seed=SEED,
    workers=1,
    updating='deferred'
)

# ---- BEST WEIGHTS ----
best_flat_weights = result.x
best_weights = reshape_weights(best_flat_weights, brain)
set_weights(brain, best_weights)
print("Best fitness from DE:", -result.fun)

# ---- VISUALIZATION ----
def visualize_policy(weights):
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    viewer = EvoViewer(env.sim)
    viewer.track_objects('robot')
    state = env.reset()[0]
    for t in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    viewer.close()
    env.close()

i = 0
while i < 5:
    visualize_policy(best_weights)
    i += 1
