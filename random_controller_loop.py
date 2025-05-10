import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from controller_neural import *
import torch

NUM_GENERATIONS = 1950
STEPS = 500
#SCENARIO = 'DownStepper-v0'
SCENARIO = 'ObstacleTraverser-v0'

robot_structure = np.array([
    [1,3,1,0,0],
    [4,1,3,2,2],
    [3,4,4,4,4],
    [3,0,0,3,2],
    [0,0,0,0,2]
])

connectivity = get_full_connectivity(robot_structure)
input_size = gym.make(SCENARIO, body=robot_structure, connections=connectivity).observation_space.shape[0]
output_size = gym.make(SCENARIO, body=robot_structure, connections=connectivity).action_space.shape[0]

brain = NeuralController(input_size, output_size)

def evaluate_fitness(weights, view=False):
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    if view:
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

    state = env.reset()[0]
    t_reward = 0
    start_pos = np.mean(sim.object_pos_at_time(0, "robot")[0])
    for t in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        if view:
            viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        t_reward += reward

        if terminated or truncated:
            break
    
    end_pos = np.mean(sim.object_pos_at_time(sim.get_time(), "robot")[0])
    distance_traveled = end_pos - start_pos

    if view:
        viewer.close()
    env.close()
    return t_reward, distance_traveled

# ---- VISUALIZATION ----
def visualize_policy(weights):
    set_weights(brain, weights)  # Load weights into the network
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]  # Get initial state
    for t in range(STEPS):  
        # Update actuation before stepping
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        action = brain(state_tensor).detach().numpy().flatten() # Get action
        viewer.render('screen') 
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
            break

    viewer.close()
    env.close()

# Run 5 independent cycles
for cycle in range(5):
    seed = 40 + cycle
    np.random.seed(seed)
    random.seed(seed)

    best_fitness = -np.inf
    best_distance = -np.inf
    best_weights = None

    print(f"\n--- Cycle {cycle+1}/5 | Seed: {seed} ---")
    for generation in range(NUM_GENERATIONS):
        random_weights = [np.random.randn(*param.shape) for param in brain.parameters()]
        fitness, distance = evaluate_fitness(random_weights)
        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = random_weights
        if distance > best_distance:
            best_distance = distance
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}, Fitness: {fitness:.5f}, Distance: {distance:.5f}")


    # Save best fitness and distance in individual file
    filename = f"task2_obstacle/Random/Random_{cycle+1}.txt"
    with open(filename, "w") as f:
        f.write(f"ROBOT: {cycle + 1}\n")
        f.write(f"Best Fitness: {best_fitness:.6f}\n")
        f.write(f"Best Distance: {best_distance:.6f}\n")
        f.write(f"Seed: {seed}\n")

    i = 0
    while i < 5:
        visualize_policy(best_weights)
        i += 1
