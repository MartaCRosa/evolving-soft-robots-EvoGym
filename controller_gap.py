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

SCENARIO = 'GapJumper-v0'
#SCENARIO = 'CaveCrawler-v0'

robot_structure = np.array([ 
[1,3,1,0,0],
[4,1,3,2,2],
[3,4,4,4,4],
[3,0,0,3,2],
[0,0,0,0,2]
])

connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
#for i in range(env.sim.get_num_objects()):
#    print(f"Object {i}: {env.sim.get_name(i)}")
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
    z_heights = []
    velocity_list = []
    airborne_frames = 0
    active_time = 0

    start_pos = np.mean(sim.object_pos_at_time(0, "robot")[0])
    z_com_prev = None

    for _ in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        if view:
            viewer.render('screen')

        state, reward, terminated, truncated, _ = env.step(action)

        com_pos = np.mean(sim.object_pos_at_time(sim.get_time(), "robot")[0])
        z_com = com_pos
        z_heights.append(z_com)

        # Estimate contact with floor using terrain elevation
        floor_obs = env.get_floor_obs("robot", ["platform_1"], sight_dist=1, sight_range=5)
        min_dist_to_floor = np.min(floor_obs)
        if min_dist_to_floor > 0.2:  # adjust threshold as needed
            airborne_frames += 1


        # Vertical velocity (only positive is jump)
        if z_com_prev is not None:
            vertical_velocity = z_com - z_com_prev
        else:
            vertical_velocity = 0
        z_com_prev = z_com

        velocities = sim.vel_at_time(sim.get_time())
        avg_x_velocity = np.mean(velocities[0])
        velocity_list.append(avg_x_velocity)

        active_time += 1

        if terminated or truncated:
            break

    end_pos = np.mean(sim.object_pos_at_time(sim.get_time(), "robot")[0])
    distance_traveled = end_pos - start_pos
    avg_velocity = np.mean(velocity_list)
    z_std = np.std(z_heights)

    # --- Fitness components ---
    distance_bonus = distance_traveled * 50
    hop_bonus = z_std * 100  # vertical movement indicates hopping
    airtime_bonus = airborne_frames * 2
    jump_speed_bonus = max(0, vertical_velocity) * 200
    fall_penalty = -200 if terminated and not truncated else 0
    stiffness_penalty = -100 if z_std < 0.1 else 0  # discourage rigid motion

    final_fitness = (
        distance_bonus +
        hop_bonus +
        airtime_bonus +
        jump_speed_bonus +
        fall_penalty +
        stiffness_penalty
    )

    if view:
        viewer.close()
    env.close()

    print(f"Distance: {distance_traveled:.2f}, Zstd: {z_std:.2f}, "
          f"Airtime: {airborne_frames}, VZ: {vertical_velocity:.2f}, "
          f"Final: {final_fitness:.2f}")

    return final_fitness, distance_traveled


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
    
    fitnesses, distances = zip(*[evaluate_fitness(reshape_weights(ind, brain)) for ind in combined])
    top_indices = np.argsort(fitnesses)[-MU:]

    population = combined[top_indices]
    best_fitness = fitnesses[top_indices[-1]]
    best_distance_in_generation = distances[top_indices[-1]]

    fitness_progress.append(best_fitness)
    print(f"Generation {generation+1}/{NUM_ITERATIONS}, Best Fitness: {best_fitness:.5f}, Best Distance: {best_distance_in_generation:.5f}")

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
