import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from controller_neural import *

# --- EvoGym Setup ---
NUM_GENERATIONS = 20
STEPS = 500
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

POPULATION_SIZE = 50
SIGMA = 0.1
LEARNING_RATE = 0.03

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
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
brain = NeuralController(input_size, output_size)

# --- Fitness Function ---
def evaluate_fitness(weights, view=False):
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    if view:
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

    state = env.reset()[0]
    total_reward = 0

    for _ in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        if view:
            viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    if view:
        viewer.close()
    env.close()
    return total_reward

# --- Weight Helpers ---
def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])

def reshape_weights(flat_weights, shapes):
    new_weights = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        new_weights.append(flat_weights[idx:idx + size].reshape(shape))
        idx += size
    return new_weights

# --- NES Optimization ---
original_weights = [np.random.randn(*p.shape) for p in brain.parameters()]
shapes = [w.shape for w in original_weights]
theta = flatten_weights(original_weights)
fitness_history = []

for gen in range(NUM_GENERATIONS):
    noise = np.random.randn(POPULATION_SIZE, theta.size)
    rewards = np.zeros(POPULATION_SIZE)

    for i in range(POPULATION_SIZE):
        weights_try = theta + SIGMA * noise[i]
        structured = reshape_weights(weights_try, shapes)
        rewards[i] = evaluate_fitness(structured)

    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
    gradient = np.dot(noise.T, rewards) / POPULATION_SIZE
    theta += LEARNING_RATE * gradient

    best_idx = np.argmax(rewards)
    best_flat = theta + SIGMA * noise[best_idx]
    best_weights = reshape_weights(best_flat, shapes)
    fitness = evaluate_fitness(best_weights)
    fitness_history.append(fitness)

    print(f"Generation {gen+1}, Fitness: {fitness:.2f}")

# --- Visualize Best Policy ---
def visualize_policy(weights):
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]
    for _ in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        viewer.render('screen')
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    viewer.close()
    env.close()

i = 0
while i < 5:
    visualize_policy(best_weights)
    i += 1