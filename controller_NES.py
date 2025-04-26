import numpy as np
import random
import torch
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from controller_neural import *
import matplotlib.pyplot as plt

# --- EvoGym Setup ---
NUM_ITERATIONS = 10
STEPS = 500
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

SCENARIO = 'DownStepper-v0'

robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])

connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

# ---- CONTROLLER ----
brain = NeuralController(input_size, output_size)

def get_weights(model):
    return [p.data.numpy() for p in model.parameters()]

def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data = torch.tensor(w, dtype=torch.float32)

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

# --- Fitness Function ---
def evaluate_fitness(weights, view=False):
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    if view:
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

    state = env.reset()[0]
    t_reward = 0
    velocity_list = []
    start_pos = np.mean(sim.object_pos_at_time(0, "robot")[0])

    active_time = 0  # New: counts how many steps robot stays alive

    for _ in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        if view:
            viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        t_reward += reward

        velocities = sim.vel_at_time(sim.get_time())
        avg_x_velocity = np.mean(velocities[0])
        velocity_list.append(avg_x_velocity)

        active_time += 1

        if terminated or truncated:
            break

    end_pos = np.mean(sim.object_pos_at_time(sim.get_time(), "robot")[0])
    distance_traveled = end_pos - start_pos
    avg_velocity = np.mean(velocity_list)

    # --- Fitness calculation ---
    distance_bonus = distance_traveled * 20 if distance_traveled > 0 else distance_traveled * 50
    velocity_bonus = avg_velocity * 30 if avg_velocity > 0 else avg_velocity * 50
    fall_penalty = -100 if terminated and not truncated else 0
    #time_bonus = active_time * 0.25  # New: reward robot for surviving longer

    final_fitness = t_reward + distance_bonus + velocity_bonus + fall_penalty #+ time_bonus

    if view:
        viewer.close()
    env.close()

    print(f"Distance: {distance_traveled:.4f}, Velocity: {avg_velocity:.4f}, Time: {active_time}, Final: {final_fitness:.4f}")

    return final_fitness

# --- NES Setup ---
ALPHA = 0.1  # Step size for NES
SIGMA = 0.1  # Mutation strength (standard deviation)
MU = 10  # Number of parents
NUM_ITERATIONS = 10

initial_weights = flatten_weights(get_weights(brain))
mean = np.zeros_like(initial_weights)
covariance = np.eye(len(initial_weights))  # Identity matrix for covariance

fitness_progress = []

for generation in range(NUM_ITERATIONS):
    # --- Generate offspring ---
    offspring = []
    for _ in range(MU):
        noise = np.random.multivariate_normal(np.zeros_like(mean), covariance)
        offspring.append(mean + SIGMA * noise)
    
    offspring = np.array(offspring)

    # --- Evaluate fitness ---
    fitnesses = np.array([evaluate_fitness(reshape_weights(ind, brain)) for ind in offspring])

    # --- Update mean and covariance ---
    # We want to select the best individuals from offspring
    best_indices = np.argsort(fitnesses)[-MU:]
    best_offspring = offspring[best_indices]

    # Update the mean and covariance based on the best performing individuals
    mean = np.mean(best_offspring, axis=0)

    # Estimate covariance from the top-performing offspring
    centered_offspring = best_offspring - mean
    covariance = np.cov(centered_offspring.T)

    best_fitness = fitnesses[best_indices[-1]]
    fitness_progress.append(best_fitness)

    print(f"Generation {generation+1}/{NUM_ITERATIONS}, Best Fitness: {best_fitness:.5f}")

# --- Plot ---
plt.figure(figsize=(10,5))
plt.plot(fitness_progress, label="Best Fitness", marker='o', color='green')
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("NES Fitness Progress")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Visualize Best Policy ---
def visualize_policy(weights):
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    viewer = EvoViewer(env.sim)
    viewer.track_objects('robot')
    state = env.reset()[0]
    for _ in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    viewer.close()
    env.close()

for _ in range(10):
    visualize_policy(mean)
