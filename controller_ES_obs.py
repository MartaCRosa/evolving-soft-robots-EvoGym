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

SCENARIO = 'ObstacleTraverser-v0'

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
    z_heights = []
    hop_reward = 0
    start_pos = np.mean(sim.object_pos_at_time(0, "robot")[0])
    active_time = 0

    for _ in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        if view:
            viewer.render('screen')

        state, reward, terminated, truncated, _ = env.step(action)
        t_reward += reward

        # --- Compute average velocity ---
        velocities = sim.vel_at_time(sim.get_time())
        avg_x_velocity = np.mean(velocities[0])
        velocity_list.append(avg_x_velocity)

        # --- Get terrain and robot z-position ---
        terrain_obs = sim.get_floor_obs("robot", ["terrain"], sight_dist=2, sight_range=5)
        z_pos = np.mean(sim.object_pos_at_time(sim.get_time(), "robot")[0][1])
        z_heights.append(z_pos)

        # --- Reward hopping if near an obstacle and robot is elevated ---
        obstacle_near = np.any(terrain_obs < 1.5)  # terrain is close below
        if obstacle_near and z_pos > 1.5:
            hop_reward += 5  # reward hopping when it's actually useful

        active_time += 1
        if terminated or truncated:
            break

    end_pos = np.mean(sim.object_pos_at_time(sim.get_time(), "robot")[0])
    distance_traveled = end_pos - start_pos
    avg_velocity = np.mean(velocity_list)

    # --- Additional hop-based bonus (based on movement style) ---
    vertical_movement = np.std(z_heights)
    hop_bonus = vertical_movement * 50  # Encourage bouncy/hopping motion

    # --- Fitness calculation ---
    distance_bonus = distance_traveled * 20 if distance_traveled > 0 else distance_traveled * 50
    velocity_bonus = avg_velocity * 50 if avg_velocity > 0 else avg_velocity * 50
    fall_penalty = -200 if terminated and not truncated else 0
    hop_penalty = -50 if vertical_movement < 0.1 else 0  # discourage stiff robots

    final_fitness = (
        t_reward +
        distance_bonus +
        velocity_bonus +
        hop_bonus +
        hop_reward +
        hop_penalty +
        fall_penalty
    )

    if view:
        viewer.close()
    env.close()

    print(f"Distance: {distance_traveled:.4f}, Velocity: {avg_velocity:.4f}, "
          f"Time: {active_time}, HopReward: {hop_reward:.2f}, Final: {final_fitness:.4f}")

    return final_fitness


# --- (mu + lambda) Evolution Strategy Setup ---
MU = 5
LAMBDA = 10
SIGMA = 0.2

initial_weights = flatten_weights(get_weights(brain))
population = np.array([initial_weights + np.random.normal(0, SIGMA, size=initial_weights.shape) for _ in range(MU)]) 
fitness_progress = []

for generation in range(NUM_ITERATIONS):
    offspring = []

    for _ in range(LAMBDA):
        parent_idx = np.random.randint(0, MU)
        parent = population[parent_idx]
        child = parent + np.random.normal(0, SIGMA, size=parent.shape)  # Each child is created by taking a parent and adding random noise
        offspring.append(child)

    offspring = np.array(offspring)
    combined = np.vstack((population, offspring))
    
    fitnesses = np.array([evaluate_fitness(reshape_weights(ind, brain)) for ind in combined])
    top_indices = np.argsort(fitnesses)[-MU:]

    population = combined[top_indices]
    best_fitness = fitnesses[top_indices[-1]]

    fitness_progress.append(best_fitness)
    print(f"Generation {generation+1}/{NUM_ITERATIONS}, Best Fitness: {best_fitness:.5f}")

# Set best weights
best_individual = population[-1]

# --- Plot ---
plt.figure(figsize=(10,5))
plt.plot(fitness_progress, label="Best Fitness", color='green')
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("(mu + lambda) ES Fitness Progress")
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
    visualize_policy(reshape_weights(best_individual, brain))
