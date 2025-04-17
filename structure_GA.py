import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from controllers_fixed import *
import matplotlib.pyplot as plt

# abordagens nao validas no 3.1
# CMA-ES
# evolução diferencial

# começar com algo simples e dps alterar pontos fracos
# vantagem representação de inteiros - 
# individuos invalidos - vai fora e gera outro/fitness negativa

# ---- PARAMETERS ----
NUM_GENERATIONS = 40 #250  # Number of generations to evolve #hyperparametro
#comecar com grelha pequena e dps explorar
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid # manter fixo para a evolução da estrutura
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500 #fixo

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

        successful = False
        #reward_list = []  # Store rewards per step
        velocity_list = []  # Store velocity per step        
        start_pos = np.mean(sim.object_pos_at_time(0, "robot")[0]) # Get initial position (center of mass)

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
            distance_bonus = distance_traveled*2  # forward movement gets rewarded proportionaly to distance
        #distance_bonus = max(distance_traveled * 1.3, 0) # reward proportional to the distance, reward = 0 if it's moving backwards

        # Velocity bonus -> FROM CHATGPT
        avg_velocity = np.mean(velocity_list)  # average x-velocity across all steps
        if avg_velocity <= 0:
            velocity_bonus = -10
        else:
            velocity_bonus = avg_velocity *3  # rewards higher average velocities, reward = 0 if velocity_bonus < 0

        # Actuator(-) bonus (interval of numbers that make sense for the environment) -> CHATGPT
        actuator_count = np.count_nonzero(robot_structure == 4)
        if 2 <= actuator_count <= 4:  #2-4 walker, 3-7 bridge
            actuator_bonus = 10  
        else:
            actuator_bonus = -5  

        # Final fitness score
        final_fitness = t_reward + distance_bonus + velocity_bonus + actuator_bonus #+ stability_penalty + leg_bonus
        print("Distance traveled: ", distance_traveled, "Average velocity: ", avg_velocity, "Final Fitness: ", final_fitness)
        return max(final_fitness, 0)

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
    
    # garantir que os pais tem a mesma forma e tamanho
    if parent1.shape != parent2.shape:
        raise ValueError("Parents must have the same shape!")

    rows, cols = parent1.shape

    for i in range(3): #vai experimentar 3 vezes crossover se nao estiver a conseguir fazer
            crossover_point = random.randint(1, rows - 1)    # escolher um ponto random de crossover
            child = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))  # chil é criado combinando as duas partes dos pais a partir do ponto de crossover selecionado
            if is_connected(child):
                return child
            
    # caso o child nao esteja conected avalia a fitness dos pais e retorna o melhor deles
    return copy.deepcopy(parent1) if evaluate_fitness(parent1) >= evaluate_fitness(parent2) else copy.deepcopy(parent2)

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
    
    return copy.deepcopy(melhor_robot)

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

        new_population = [copy.deepcopy(robot) for robot in population[:ELITISM_COUNT]]

        while len(new_population) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
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

    # Save the plot to a file
    plot_filename = f'task1_walker/GA/40gen_fitness_progression_run_{i+1}.png'  # Save the plot as a .png file
    plt.savefig(plot_filename)

    # Close the plot
    plt.close()


    return best_robot, best_fitness

# ----- RUN LOOP --------- #

POP_SIZE = 20
MUTATION_RATE = 0.2
NUM_RUNS = 5

for i in range(NUM_RUNS):
    print(f"\n--- Starting Run {i+1} ---")
    best_robot, best_fitness = genetic_algorithm(POP_SIZE, MUTATION_RATE)
    print(f"Run {i+1} Best Robot:")
    print(best_robot)
    print("Best Fitness:", best_fitness)

    # Save structure and fitness to txt
    txt_filename = f'task1_walker/GA/GA_40gen_{i+6}.txt'
    with open(txt_filename, 'w') as f:
        f.write(f"RUN {i+1}\n")
        f.write(f"Best Fitness: {best_fitness}\n")
        f.write("Best robot structure:\n")
        f.write(str(best_robot))

    # Simulate and create gif
    utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
    gif_path = f'task1_walker/GA/GA_40gen_{i+1}.gif'
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
