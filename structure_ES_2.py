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
NUM_GENERATIONS = 50 #250  # Number of generations to evolve
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
        reward_list = []  # Store rewards per step

        for t in range(STEPS):  
            actuation = CONTROLLER(action_size, t)
            if view:
                viewer.render('screen')

            ob, reward, terminated, truncated, info = env.step(actuation)  
            t_reward += reward  
            reward_list.append(reward)

            if terminated:
                successful = True
                break  
            if truncated:
                ran_out_of_time = True
                break

        viewer.close()
        env.close()

        # === FITNESS CALCULATION ===
        speed_score = t_reward / STEPS  

        # Actuator bonus (encourage but not too many)
        actuator_count = np.count_nonzero(robot_structure == 4)
        if 2 <= actuator_count <= 6:
            actuator_bonus = 20  
        else:
            actuator_bonus = -10 * abs(actuator_count - 4)  

        # Running out of time penalty
        if ran_out_of_time:
            t_reward *= 0.8  

        # Stability penalty (chaotic movement)
        reward_variance = np.var(reward_list)
        stability_penalty = -10 if reward_variance > 5 else 0  

        # Final fitness score
        final_fitness = speed_score * 100 + actuator_bonus + stability_penalty
        return max(final_fitness, 0)  

    except (ValueError, IndexError):
        return 0.0



# ---- INITIALIZATION ----
def create_random_robot():
    """Generate a valid random robot structure."""
    
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size) #podemos criar nós o robot
    return random_robot


def mutate_robot(parent, mutation_rate=0.2):
    """Mutate the structure by changing multiple voxel types."""
    child = copy.deepcopy(parent)
    num_voxels = child.shape[0] * child.shape[1]
    num_mutations = max(1, int(mutation_rate * num_voxels))  # At least one mutation
    
    for _ in range(num_mutations):
        x, y = np.random.randint(0, child.shape[0]), np.random.randint(0, child.shape[1])
        new_voxel = random.choice([v for v in VOXEL_TYPES if v != child[x, y]])  # Ensure change
        child[x, y] = new_voxel

    return child if is_connected(child) else parent  # Ensure connectivity

def crossover(parent1, parent2):
    """Combine two parents using random crossover."""
    child = copy.deepcopy(parent1)
    mask = np.random.rand(*parent1.shape) > 0.5  # 50% chance to take from parent2
    child[mask] = parent2[mask]
    return child if is_connected(child) else parent1  # Ensure connectivity

def evolution_strategy():
    """Perform (μ + λ) Evolution Strategies with crossover and enhanced mutation."""
    population = [create_random_robot() for _ in range(MU)]
    fitness_scores = [evaluate_fitness(robot) for robot in population]

    best_fitness_over_time = []
    avg_fitness_over_time = []

    for gen in range(NUM_GENERATIONS):
        offspring = []

        # Generate offspring through mutation & crossover
        for _ in range(LAMBDA):
            if random.random() < 0.5:  # 50% mutation, 50% crossover
                parent = random.choice(population)
                child = mutate_robot(parent)
            else:
                p1, p2 = random.sample(population, 2)
                child = crossover(p1, p2)
                
            offspring.append(child)

        # Evaluate offspring fitness
        offspring_fitness = [evaluate_fitness(robot) for robot in offspring]

        # (μ, λ) selection: Only select from offspring
        sorted_indices = np.argsort(offspring_fitness)[::-1]  
        population = [offspring[i] for i in sorted_indices[:MU]]
        fitness_scores = [offspring_fitness[i] for i in sorted_indices[:MU]]

        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)

        best_fitness_over_time.append(best_fitness)
        avg_fitness_over_time.append(avg_fitness)

        print(f"Generation {gen+1}: Best Fitness = {best_fitness}, Avg Fitness = {avg_fitness}")

    # Plot fitness vs. generations
    plt.figure(figsize=(8, 5))
    plt.plot(range(NUM_GENERATIONS), best_fitness_over_time, label="Best Fitness", color="blue")
    plt.plot(range(NUM_GENERATIONS), avg_fitness_over_time, label="Average Fitness", color="orange", linestyle="dashed")
    
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Evolution Strategy: Fitness Progression")
    plt.legend()
    plt.grid()
    plt.show()

    return population[np.argmax(fitness_scores)], max(fitness_scores)

MU = 5  
LAMBDA = 15  
best_robot, best_fitness = evolution_strategy()
print("Best robot structure found:")
print(best_robot)
print("Best fitness score:", best_fitness)

i = 0
while i < 5:
    utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
    i += 1
utils.create_gif(best_robot, filename='gifs/ES_250gen_500step_new.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)