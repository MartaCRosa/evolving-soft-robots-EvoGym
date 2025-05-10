import numpy as np
import random
import copy
import torch
import gymnasium as gym
from evogym.envs import *
from evogym import get_full_connectivity, EvoViewer, sample_robot, is_connected
from controller_neural import NeuralController
import matplotlib.pyplot as plt

# --- Parameters ---
NUM_GENERATIONS = 10
MU = 5
LAMBDA = 10
STEPS = 500
SIGMA = 0.2
SCENARIO = 'GapJumper-v0'

# --- EvoGym Setup ---
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
VOXEL_TYPES = [0, 1, 2, 3, 4]

# --- Helper Functions ---
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

def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data = torch.tensor(w, dtype=torch.float32)

def mutate_controller_weights(weights, sigma=0.2):
    return weights + np.random.normal(0, sigma, size=weights.shape) # mutates controller weights by adding gaussian noise

def create_random_robot():
    grid_size = (random.randint(*MIN_GRID_SIZE), random.randint(*MAX_GRID_SIZE))
    robot, _ = sample_robot(grid_size)
    return robot

def create_controller_for_robot(robot):
    conn = get_full_connectivity(robot)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot, connections=conn)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    env.close()

    controller = NeuralController(input_size, output_size)
    initial_weights = flatten_weights([p.data.numpy() for p in controller.parameters()])
    return initial_weights, input_size, output_size

def mutate_robot(robot):
    for _ in range(3):
        child = copy.deepcopy(robot)
        x, y = np.random.randint(0, child.shape[0]), np.random.randint(0, child.shape[1])
        new_voxel = random.choice([v for v in VOXEL_TYPES if v != child[x, y]])
        child[x, y] = new_voxel
        if is_connected(child):
            return child
    return robot

def evaluate(robot, weights, input_size, output_size, view=False):
    try:
        conn = get_full_connectivity(robot)
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot, connections=conn)
        sim = env.sim
        controller = NeuralController(input_size, output_size)
        set_weights(controller, weights)

        if view:
            viewer = EvoViewer(sim)
            viewer.track_objects('robot')

        state = env.reset()[0]
        t_reward = 0
        velocity_list = []
        vertical_velocity_list = []
        start_pos = np.mean(sim.object_pos_at_time(0, "robot")[0])
        active_time = 0

        for _ in range(STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = controller(state_tensor).detach().numpy().flatten()

            if view:
                viewer.render('screen')

            state, reward, terminated, truncated, _ = env.step(action)
            t_reward += reward

            velocities = sim.vel_at_time(sim.get_time())
            avg_x_velocity = np.mean(velocities[0])
            avg_y_velocity = np.mean(velocities[1])

            velocity_list.append(avg_x_velocity)
            vertical_velocity_list.append(abs(avg_y_velocity))

            active_time += 1
            if terminated or truncated:
                break

        end_pos = np.mean(sim.object_pos_at_time(sim.get_time(), "robot")[0])
        distance_traveled = end_pos - start_pos
        avg_velocity = np.mean(velocity_list)
        avg_vertical_velocity = np.mean(vertical_velocity_list)

        distance_bonus = distance_traveled * 20 if distance_traveled > 0 else distance_traveled * 50
        velocity_bonus = avg_velocity * 50
        hop_bonus = avg_vertical_velocity * 60
        fall_penalty = -200 if terminated and not truncated else 0

        final_fitness = t_reward + distance_bonus + velocity_bonus + hop_bonus + fall_penalty

        if view:
            print(f"Distance: {distance_traveled:.4f}, Velocity: {avg_velocity:.4f}, Vertical Vel: {avg_vertical_velocity:.4f}, Time: {active_time}, Final: {final_fitness:.4f}")
            viewer.close()
        env.close()

        return final_fitness

    except Exception as e:
        print("Evaluation error:", e)
        return 0.0

import os

def run_experiment(run_id, num_generations, save_gif=False):
    robot_population = []
    controller_population = []
    input_output_sizes = []
    fitness_history = []

    for _ in range(MU):
        robot = create_random_robot()
        controller, input_size, output_size = create_controller_for_robot(robot)
        robot_population.append(robot)
        controller_population.append(controller)
        input_output_sizes.append((input_size, output_size))

    for gen in range(num_generations):
        print(f"\n[Run {run_id}] --- Generation {gen+1} ---")

        fitness_scores = []
        for robot, controller, (input_size, output_size) in zip(robot_population, controller_population, input_output_sizes):
            weights = reshape_weights(controller, NeuralController(input_size, output_size))
            fitness = evaluate(robot, weights, input_size, output_size)
            print(f"Fitness: {fitness}")
            fitness_scores.append(fitness)

        robot_offspring = []
        controller_offspring = []
        offspring_ios = []

        for _ in range(LAMBDA):
            idx = random.randint(0, MU - 1)
            parent_robot = robot_population[idx]
            parent_controller = controller_population[idx]
            input_size, output_size = input_output_sizes[idx]

            child_robot = mutate_robot(parent_robot)

            conn = get_full_connectivity(child_robot)
            env = gym.make(SCENARIO, max_episode_steps=STEPS, body=child_robot, connections=conn)
            child_input_size = env.observation_space.shape[0]
            child_output_size = env.action_space.shape[0]
            env.close()

            if (child_input_size == input_size) and (child_output_size == output_size): 
                child_controller = mutate_controller_weights(np.copy(parent_controller), sigma=SIGMA)
            else:
                child_controller, child_input_size, child_output_size = create_controller_for_robot(child_robot)

            robot_offspring.append(child_robot)
            controller_offspring.append(child_controller)
            offspring_ios.append((child_input_size, child_output_size))

        offspring_fitness = []
        for robot, controller, (input_size, output_size) in zip(robot_offspring, controller_offspring, offspring_ios):
            weights = reshape_weights(controller, NeuralController(input_size, output_size))
            fitness = evaluate(robot, weights, input_size, output_size)
            print(f"Fitness: {fitness}")
            offspring_fitness.append(fitness)

        combined_robots = robot_population + robot_offspring
        combined_controllers = controller_population + controller_offspring
        combined_ios = input_output_sizes + offspring_ios
        combined_fitness = fitness_scores + offspring_fitness

        top_idx = np.argsort(combined_fitness)[-MU:]
        robot_population = [combined_robots[i] for i in top_idx]
        controller_population = [combined_controllers[i] for i in top_idx]
        input_output_sizes = [combined_ios[i] for i in top_idx]

        best_fitness = max(combined_fitness)
        fitness_history.append(best_fitness)
        print(f"Best Fitness: {best_fitness:.4f}")

    # --- Save Plot ---
    os.makedirs("results", exist_ok=True)
    plt.figure()
    plt.plot(fitness_history, label=f'Run {run_id}')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Fitness Evolution - Run {run_id}")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/fitness_run_{run_id}.png")
    plt.close()

    # --- Save GIF ---
    if save_gif:
        best_robot = robot_population[-1]
        best_controller = controller_population[-1]
        input_size, output_size = input_output_sizes[-1]
        weights = reshape_weights(best_controller, NeuralController(input_size, output_size))
        evaluate(best_robot, weights, input_size, output_size, view=True)  # Will auto-save if EvoViewer handles it

# --- Run 5 Experiments ---
for run in range(1, 6):
    run_experiment(run_id=run, num_generations=NUM_GENERATIONS, save_gif=True)

