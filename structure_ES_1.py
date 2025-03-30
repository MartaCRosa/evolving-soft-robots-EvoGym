import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from controllers_fixed import *
import matplotlib.pyplot as plt

# CONSISTENTE E RAPIDO
# O MSM NUMERO DE INDIVIDUOS


# ---- PARAMETERS ----
NUM_GENERATIONS = 250 #250  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500

#SCENARIO = 'Walker-v0'
SCENARIO = 'BridgeWalker-v0' #dá jeito ter atuadores para bridge

# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

CONTROLLER = alternating_gait
#CONTROLLER = sinusoidal_wave
#CONTROLLER = hopping_motion


# ---- EVALUATION FUNCTION ----
def evaluate_fitness(robot_structure, view=False):    
    try:
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        t_reward = 0
        action_size = sim.get_dim_action_space('robot')
        successful = False
        ran_out_of_time = False

        for t in range(STEPS):  
            actuation = CONTROLLER(action_size, t)
            if view:
                viewer.render('screen')

            ob, reward, terminated, truncated, info = env.step(actuation)  
            t_reward += reward  

            if terminated:
                successful = True
                break  

            if truncated:
                ran_out_of_time = True
                break

        viewer.close()
        env.close()

        # Reward actuators (active voxels) 
        #actuator_bonus = np.count_nonzero(robot_structure == 4)   

        # Modify fitness based on termination type
        if successful:
            t_reward += 500 - t  # Bonus for reaching goal quickly
        elif ran_out_of_time:
            t_reward *= 0.9  # Penalty for taking too long

        return t_reward #+ actuator_bonus  # Final fitness
    except (ValueError, IndexError):
        return 0.0


# ---- INITIALIZATION ----
def create_random_robot():
    """Generate a valid random robot structure."""
    
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size) #podemos criar nós o robot
    return random_robot


# ---- MUTATION FUNCTION ----
def mutate_robot(parent):
    """Mutate the structure by changing a random voxel type."""
    child = copy.deepcopy(parent)
    x, y = np.random.randint(0, child.shape[0]), np.random.randint(0, child.shape[1])
    new_voxel = random.choice([v for v in VOXEL_TYPES if v != child[x, y]])  # Ensure mutation occurs
    child[x, y] = new_voxel
    return child if is_connected(child) else parent  # Ensure connectivity


# ---- EVOLUTION STRATEGIES (μ + λ) ---- 
def evolution_strategy():
    """Perform (μ + λ) Evolution Strategies optimization and track fitness."""
    population = [create_random_robot() for _ in range(MU)]
    fitness_scores = [evaluate_fitness(robot) for robot in population]
    
    best_fitness_over_time = []  # Track best fitness per generation
    avg_fitness_over_time = []   # Track average fitness per generation

    for gen in range(NUM_GENERATIONS):
        offspring = [mutate_robot(random.choice(population)) for _ in range(LAMBDA)]
        offspring_fitness = [evaluate_fitness(robot) for robot in offspring]

        combined_population = population + offspring
        combined_fitness = fitness_scores + offspring_fitness
        sorted_indices = np.argsort(combined_fitness)[::-1]  

        population = [combined_population[i] for i in sorted_indices[:MU]]
        fitness_scores = [combined_fitness[i] for i in sorted_indices[:MU]]

        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)

        best_fitness_over_time.append(best_fitness)
        avg_fitness_over_time.append(avg_fitness)

        print(f"Generation {gen+1}: Best Fitness = {best_fitness}, Avg Fitness = {avg_fitness}")

    # Plot fitness vs. generations
    plt.figure(figsize=(8, 5))
    plt.plot(range(NUM_GENERATIONS), best_fitness_over_time, label="Best Fitness", color="blue")
    #plt.plot(range(NUM_GENERATIONS), avg_fitness_over_time, label="Average Fitness", color="orange", linestyle="dashed")
    
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Evolution Strategy: Fitness Progression")
    plt.legend()
    plt.grid()
    plt.show()

    return population[np.argmax(fitness_scores)], max(fitness_scores)


# (λ = 2μ) or (λ = 3μ) or (λ = 7μ)
MU = 5  # Number of parents
LAMBDA = 15  # Number of offspring
best_robot, best_fitness = evolution_strategy()
print("Best robot structure found:")
print(best_robot)
print("Best fitness score:", best_fitness)

i = 0
while i < 5:
    utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
    i += 1
utils.create_gif(best_robot, filename='task1_bridge/ES_1.1/gif/ES_250gen_500steps.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)