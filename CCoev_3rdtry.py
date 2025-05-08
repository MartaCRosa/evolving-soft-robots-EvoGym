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
def create_random_robot():
    grid_size = (random.randint(*MIN_GRID_SIZE), random.randint(*MAX_GRID_SIZE))
    robot, _ = sample_robot(grid_size)
    return robot

def mutate_robot(robot, max_attempts=5):
    for _ in range(max_attempts):
        child = copy.deepcopy(robot)
        x, y = np.random.randint(0, child.shape[0]), np.random.randint(0, child.shape[1])
        new_voxel = random.choice([v for v in VOXEL_TYPES if v != child[x, y]])
        child[x, y] = new_voxel
        if is_connected(child):
            return child
    return robot  # fallback

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

def evaluate(robot, weights, view=False):
    try:
        conn = get_full_connectivity(robot)
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot, connections=conn)
        sim = env.sim
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
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

        # --- Fitness calculation ---
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


# --- Initialization ---
print("Initializing robot population...")
robot_population = [create_random_robot() for _ in range(MU)]

# Get controller dimensions from a real evaluation
print("Determining controller shape from sample robot...")
valid_robot = None
for robot in robot_population:
    try:
        conn = get_full_connectivity(robot)
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot, connections=conn)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        env.close()
        valid_robot = robot
        break
    except Exception as e:
        continue

if valid_robot is None:
    raise RuntimeError("Could not create a valid robot for controller initialization.")

controller_template = NeuralController(input_size, output_size)
initial_weights = flatten_weights([p.data.numpy() for p in controller_template.parameters()])
controller_population = [initial_weights + np.random.normal(0, SIGMA, size=initial_weights.shape) for _ in range(MU)]

fitness_history = []

# --- Coevolution Loop ---
for gen in range(NUM_GENERATIONS):
    print(f"\n--- Generation {gen+1} ---")
    
    # --- Evaluate current population ---
    fitness_scores = []
    for robot in robot_population:
        controller = random.choice(controller_population)
        try:
            fitness = evaluate(robot, reshape_weights(controller, controller_template))
        except Exception as e:
            print("Evaluation error:", e)
            fitness = 0.0
        fitness_scores.append(fitness)

    # --- Generate offspring ---
    robot_offspring = [mutate_robot(random.choice(robot_population)) for _ in range(LAMBDA)]
    controller_offspring = [
        random.choice(controller_population) + np.random.normal(0, SIGMA, size=initial_weights.shape)
        for _ in range(LAMBDA)
    ]

    # --- Evaluate offspring cooperatively ---
    robot_offspring_fitness = []
    for r in robot_offspring:
        try:
            fitness = evaluate(r, reshape_weights(random.choice(controller_population), controller_template))
        except Exception as e:
            print("Robot offspring eval error:", e)
            fitness = 0.0
        robot_offspring_fitness.append(fitness)

    controller_offspring_fitness = []
    for c in controller_offspring:
        try:
            fitness = evaluate(random.choice(robot_population), reshape_weights(c, controller_template))
        except Exception as e:
            print("Controller offspring eval error:", e)
            fitness = 0.0
        controller_offspring_fitness.append(fitness)

    # --- Selection ---
    combined_robots = robot_population + robot_offspring
    combined_robot_fitness = fitness_scores + robot_offspring_fitness
    top_robot_idx = np.argsort(combined_robot_fitness)[-MU:]
    robot_population = [combined_robots[i] for i in top_robot_idx]

    combined_controllers = controller_population + controller_offspring
    combined_controller_fitness = fitness_scores + controller_offspring_fitness
    top_controller_idx = np.argsort(combined_controller_fitness)[-MU:]
    controller_population = [combined_controllers[i] for i in top_controller_idx]

    best_fitness = max(fitness_scores)
    fitness_history.append(best_fitness)
    print(f"Best Fitness: {best_fitness:.4f}")

# --- Plot Results ---
plt.plot(fitness_history, label='Best Fitness')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Cooperative Coevolution Progress")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# --- Final Visualization ---
best_robot = robot_population[-1]
best_controller = controller_population[-1]
for _ in range(3):
    evaluate(best_robot, reshape_weights(best_controller, controller_template), view=True)
