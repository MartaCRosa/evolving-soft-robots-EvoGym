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
NUM_GENERATIONS = 50 #250  # Number of generations to evolve # hiperparametro
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid # manter fixo para a evolução da estrutura
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500 

#SCENARIO = 'Walker-v0'
SCENARIO = 'BridgeWalker-v0' 

# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-) #nao mexer

CONTROLLER = alternating_gait
#CONTROLLER = sinusoidal_wave
#CONTROLLER = hopping_motion


# ---- INITIALIZATION ----
def create_random_robot():
    """Generate a valid random robot structure."""
    
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size) #podemos criar nós o robot
    return random_robot

# ---- GENETIC ALGORITHM (GA) IMPLEMENTATION ---

# Crossover Function
"""
-> This function combines (crossovers) two given robots (parents) by slicing them horizontally at a random row
and combining the top part of robot1 (parent1) with the bottom part of robot2 (parent2) together, resulting in one child.

Parameters:
parent1 - robot chosen from population to be a parent of the resulting crossedover child
parent2 - different robot from parent1 also chosen from population to be the second parent for the crossover

In order to avoid resulting disconnected robots from crossover, this function tries to perform crossover 3 times.
If after 3 attempts the child is still not connected the function will return a copy of the parent whose fitness is best

"""
def crossover(parent1, parent2):
    
    # Check if robots have the same shape to avoid conflicting crossover
    if parent1.shape != parent2.shape:
        raise ValueError("Parents must have the same shape!")

    rows, cols = parent1.shape # Gets the rows and cols reference from parent1

    # Loop that tries to effectively do crossover 3 times
    for i in range(3): 
            crossover_point = random.randint(1, rows - 1) # Random row is chosen to be the crossover point
            child = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))  # The child is the combination of the top part of parent1 and the bottom part of parent2
            if is_connected(child):
                return child # Returns child in case it is connected
            
    # In case the child is not connected, the fitness of both parents is evaluated and a copy of the parent with best fitness is returned instead
    return copy.deepcopy(parent1) if evaluate_fitness(parent1) >= evaluate_fitness(parent2) else copy.deepcopy(parent2)


# Mutation Function
"""
-> This Function mutates robots by changing randomly the type of a random voxel.
For this evolutionary algorithm, mutation only happens for a given mutation rate. If a generated random float value
is inferior to the set mutation rate, mutation occurs.

Parameters:
parent - robot to be mutated
mutation_rate - float value of mutation rate under which mutation occurs

The voxel to be mutated is chosen randomly. 
The robot mutated is returned as a result of this function only if the robot is connected. If not
the original parent is returned instead.
"""

def mutate_robot(parent, mutation_rate):
    
    if random.random() < mutation_rate:  # Check mutation probability
        child = copy.deepcopy(parent) # Copies the parent to ensure it is not the parent being mutated, but a newly generated robot 
        x, y = np.random.randint(0, child.shape[0]), np.random.randint(0, child.shape[1]) # Random chosen voxel to be mutated
        new_voxel = random.choice([v for v in VOXEL_TYPES if v != child[x, y]])  # Random choice of voxel type that will replace the mutated voxel and ensures the chosen type is not the same as the voxel being mutated
        child[x, y] = new_voxel 
        return child if is_connected(child) else parent  # Ensure connectivity
    
    return parent # If the mutated child is not connected the parent is returned in place

# Tournament selection function
"""
-> This function is intended for the selection of robot parents that will be crossedover to generate a child for the next offspring.
Selection of parents happens in a tournament: k random robots are selected out of the current generation population and 
from them, the robot with best fitness will be returned as a parent.

Parameters:
population - the population of the current generation from which parents will be chosen
k - number of randomly selected individuals for tournament from the population
"""

def tournament_selection(population, k=5):

    selected_robots = random.sample(population,k) # Random selection of k robots from the population
    fitness_scores = [(robot,evaluate_fitness(robot)) for robot in selected_robots] # Evaluation of the selected robots

    robots_sorted = sorted(fitness_scores, key=lambda x: x[1], reverse=True) # Sorting of the selected robots' fitnesses

    best_robot = robots_sorted[0][0] # Selection of the best robot
    
    return copy.deepcopy(best_robot) # Returns a copy of the best choosen robot

# Genetic algorithm function
"""
-> This function cointains the logic of the Genetic Algoritm:
This Genetic Algorithm works by using selection with elitism, crossover and mutation.
The first generation population is generated randomly. The next generations' populations are generated in the following way:
Keeping the 2 best robots (elitism) of the current generation and filling the rest offspring with robots that resulted from crossover and mutation.
Crossover happens always and the parents of crossover are selected through tournament selection. The generated child undergoes mutation
given a certain mutation rate.

Parameters:
pop_size - size of robots population of every generation
mutatio_rate - rate under which mutation will occur

The returned result of this function is the best robot out off all generation and the corresponding fitness
"""

def genetic_algorithm(pop_size, mutation_rate):

    best_robot = None
    best_fitness = -float('inf')

    ELITISM_COUNT = 2

    population = [create_random_robot() for _ in range(pop_size)] # Inicialization of population
    fitness_scores = [evaluate_fitness(robot) for robot in population] # Evaluation of the fitnesses of the population initialization

    best_fitness_over_time = [] # To store the fitness of the best robots over generations
    avg_fitness_over_time = [] # To store the average fitness for each generation

    for it in range(NUM_GENERATIONS):
        population, fitness_scores = zip(*sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)) # Every generation of the GA starts by paring the robots of that generation population with their fitness and sorting those pairs in ascending order of fitness

        if fitness_scores[0] == 0:  # If the best fitness is zero the algorithm stops running
            break 

        new_population = [copy.deepcopy(robot) for robot in population[:ELITISM_COUNT]] # The population of the next generation is initiazed with the top 2 robots of the current generation population unchanged

        while len(new_population) < pop_size: # Loop that will fill the next generation population by creating an offspring resultinf from crossover and mutation
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = crossover(p1, p2)
            child = mutate_robot(child, mutation_rate)
            new_population.append(child)

        population = new_population # The current generation's population is replaced with the population generated through elitism, crossover and mutation
        fitness_scores = [evaluate_fitness(robot) for robot in population]  # Fitnesses are recalculated in order to correspond to the new population

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

    # Save the plot to a file
    plot_filename = f'task1_walker/GA/50gen__bridge_fitness_progression_run_{i+3}.png'  # Save the plot as a .png file
    plt.savefig(plot_filename)

    # Close the plot
    plt.close()


    return best_robot, best_fitness


# ---- EVALUATION FUNCTION ----
def evaluate_fitness(robot_structure, view=False):    
    try:
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity) # Creation of the environment in which the robot will be evaluated 
        env.reset()
        sim = env.sim 
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        t_reward = 0
        action_size = sim.get_dim_action_space('robot')  

        successful = False
        #reward_list = []  # Store rewards per step
        velocity_list = []  # Store velocity per step        
        start_pos = np.mean(sim.object_pos_at_time(0, "robot")[0]) # Get initial position (center of mass)
        orientations = []

        for t in range(STEPS):  
            # Update actuation before stepping
            actuation = CONTROLLER(action_size,t)
            if view:
                viewer.render('screen')  #modo view para por no ecra, podemos nao fazer
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward
            #reward_list.append(reward)

           # Get velocity at this timestep
            velocities = sim.vel_at_time(sim.get_time())  # (2, n) array (x, y velocities)
            avg_x_velocity = np.mean(velocities[0])  # x-velocities of n point masses
            velocity_list.append(avg_x_velocity)
            orientation = sim.object_orientation_at_time(sim.get_time(),"robot")
            orientations.append(orientation)

            if terminated:
                successful = True
                break  
            if truncated:
                break
        
        end_pos = np.mean(sim.object_pos_at_time(sim.get_time(), "robot")[0])  # Get final position (center of mass)
        
        viewer.close()
        env.close()

        #-- FITNESS CALCULATION --

        # Finnishing simulation bonus
        if successful:
            t_reward *= 2  

        # Distance traveled bonus
        distance_traveled = end_pos - start_pos 
        if distance_traveled <= 0:
            distance_bonus = -20  # negative penalty to discourage backwards movement
        else:
            distance_bonus = distance_traveled*3  # forward movement gets rewarded proportionaly to distance
        #distance_bonus = max(distance_traveled * 1.3, 0) # reward proportional to the distance, reward = 0 if it's moving backwards

        # Velocity bonus -> FROM CHATGPT
        avg_velocity = np.mean(velocity_list)  # average x-velocity across all steps
        if avg_velocity <= 0:
            velocity_bonus = -10
        else:
            velocity_bonus = avg_velocity *3  # rewards higher average velocities, reward = 0 if velocity_bonus < 0

        # Actuator(-) bonus (interval of numbers that make sense for the environment) -> CHATGPT
        actuator_count = np.count_nonzero(robot_structure == 4)
        if 3 <= actuator_count <= 7:  #2-4 walker, 3-7 bridge
            actuator_bonus = 10  
        else:
            actuator_bonus = -5  

        # DO CHAT
        # Convert to numpy array
        orientation_changes = np.diff(orientations)
        orientation_range = np.max(orientations) - np.min(orientations)
        avg_change = np.mean(np.abs(orientation_changes))

        # Smart stability penalty
        stability_penalty = (orientation_range * 1.0 + avg_change * 1.0)
        stability_penalty = min(stability_penalty, 10)  # soft cap
        #as vezes a robots bons que caiem ao inicio por isso nao penalizar assim tanto
        # quanto maior a mudança de orientação maior a penalização

        # Final fitness score
        final_fitness = t_reward + distance_bonus + velocity_bonus + actuator_bonus - stability_penalty #+ stability_penalty + leg_bonus
        print("Final Fitness: ", final_fitness)
        return max(final_fitness, 0)

    except (ValueError, IndexError) as e:
        return 0.0




# ----- RUN LOOP --------- #

POP_SIZE = 20
MUTATION_RATE = 0.2
NUM_RUNS = 2

for i in range(NUM_RUNS):
    print(f"\n--- Starting Run {i+1} ---")
    best_robot, best_fitness = genetic_algorithm(POP_SIZE, MUTATION_RATE)
    print(f"Run {i+1} Best Robot:")
    print(best_robot)
    print("Best Fitness:", best_fitness)

    # Save structure and fitness to txt
    txt_filename = f'task1_walker/GA/GA_50gen_bridge{i+1}.txt'
    with open(txt_filename, 'w') as f:
        f.write(f"RUN {i+1}\n")
        f.write(f"Best Fitness: {best_fitness}\n")
        f.write("Best robot structure:\n")
        f.write(str(best_robot))

    # Simulate and create gif
    utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
    gif_path = f'task1_walker/GA/GA_50gen_bridge{i+1}.gif'
    utils.create_gif(best_robot, filename=gif_path, scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)



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
