import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from controllers_fixed import *
import matplotlib.pyplot as plt


# ---- PARAMETERS ----
NUM_GENERATIONS = 150 #250  # Number of generations to evolve
#comecar com grelha pequena e dps explorar
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500

SCENARIO = 'Walker-v0'
#SCENARIO = 'BridgeWalker-v0' #dá jeito ter atuadores para bridge
#cão ao contrario e sapo

# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-) #nao mexer

CONTROLLER = alternating_gait
#CONTROLLER = sinusoidal_wave
#CONTROLLER = hopping_motion


# ---- EVALUATION FUNCTION ----
def evaluate_fitness(robot_structure, view=False):    
    try:
        connectivity = get_full_connectivity(robot_structure)
  
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity) #gym cria cenario 
        env.reset()
        sim = env.sim  # criar cenario de raiz e por la o robot
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        t_reward = 0
        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        for t in range(STEPS):  
            # Update actuation before stepping
            actuation = CONTROLLER(action_size,t)
            if view:
                viewer.render('screen')  #modo view para por no ecra, podemos nao fazer
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward

            if terminated:
                #t_reward *=1.05
                print("Sucess! Simulation terminated.",t_reward)
                env.reset()
                break
            elif truncated:
                #t_reward *=0.9
                print("Time limit reached. Simulation truncated.",t_reward)
                env.reset()
                break

        viewer.close()
        env.close()
        return t_reward
    except (ValueError, IndexError) as e:
        return 0.0


# ---- INITIALIZATION ----
def create_random_robot():
    """Generate a valid random robot structure."""
    
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size) #podemos criar nós o robot
    return random_robot


# Algoritmo genetico


def crossover(parent1, parent2):
    """Perform crossover between two parent robots (grid-based)."""
    # Ensure parents have the same grid size
    if parent1.shape != parent2.shape:
        raise ValueError("Parents must have the same shape!")

    rows, cols = parent1.shape

    # Choose a random crossover point
    crossover_point = random.randint(1, rows - 1)  

    # Create child by combining parts from both parents
    child = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))

    return child

def mutate_robot(parent,mutation_rate):
    """Mutate the structure by changing a random voxel type."""

    if random.random() < mutation_rate:  # Check mutation probability
        child = copy.deepcopy(parent)
        x, y = np.random.randint(0, child.shape[0]), np.random.randint(0, child.shape[1])
        new_voxel = random.choice([v for v in VOXEL_TYPES if v != child[x, y]])  # Ensure mutation occurs
        child[x, y] = new_voxel
        return child if is_connected(child) else parent  # Ensure connectivity
    
    return parent # se nao mutar devolve o original

def tournament_selection(population, k=5):

    robots_selecionados = random.sample(population,k)
    fitness_scores = [(robot,evaluate_fitness(robot)) for robot in robots_selecionados]

    robots_sorted = sorted(fitness_scores, key=lambda x: x[1], reverse=True)

    melhor_robot = robots_sorted[0][0]
    
    return melhor_robot

def genetic_algorithm(pop_size,mutation_rate):

    best_robot = None
    best_fitness = -float('inf')

    ELITISM_COUNT = 2

    population = [create_random_robot() for _ in range(pop_size)]
    fitness_scores = [evaluate_fitness(robot) for robot in population]

    best_fitness_over_time = []
    avg_fitness_over_time = []

    for it in range(NUM_GENERATIONS):
        population, fitness_scores = zip(*sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True))

        if fitness_scores[0] == 0:
            break

        new_population = list(population[:ELITISM_COUNT]) # nova população começa com os 2 melhores robots

        while len(new_population) < pop_size:

            while len(new_population) < pop_size:
                p1 = tournament_selection(population)
                p2 = tournament_selection(population)
                child = crossover(p1, p2)
                child = mutate_robot(child, mutation_rate)
                new_population.append(child)

                child = crossover(p1, p2)
                child = mutate_robot(child, mutation_rate)
                new_population.append(child)

        population = new_population
        fitness_scores = [evaluate_fitness(robot) for robot in population]  # Recompute fitness

        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)


        best_fitness_over_time.append(best_fitness)
        avg_fitness_over_time.append(avg_fitness)


        print(f"Iteration {it + 1}: Best Fitness = {max(fitness_scores)}")

    best_index = np.argmax(fitness_scores)
    best_robot = population[best_index]
    best_fitness = fitness_scores[best_index]

    # Plot fitness vs. generations
    plt.figure(figsize=(8, 5))
    plt.plot(range(NUM_GENERATIONS), best_fitness_over_time, label="Best Fitness", color="blue")
    plt.plot(range(NUM_GENERATIONS), avg_fitness_over_time, label="Average Fitness", color="orange", linestyle="dashed")
    
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Genetic Algorithm: Fitness Progression")
    plt.legend()
    plt.grid()
    plt.show()

    return best_robot, best_fitness


best_robot, best_fitness = genetic_algorithm(8,0.3)
print("Best robot structure found:")
print(best_robot)
print("Best fitness score:")
print(best_fitness)
i = 0
while i < 5:
    utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
    i += 1
utils.create_gif(best_robot, filename='task1_walker/GA/GA_20gen1.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)


# Random 
#best_robot, best_fitness = random_search()
#print("Best robot structure found:")
#print(best_robot)
#print("Best fitness score:")
#print(best_fitness)
#i = 0
#while i < 5:
   # utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
  #  i += 1
#utils.create_gif(best_robot, filename='gifs/random_1.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)
