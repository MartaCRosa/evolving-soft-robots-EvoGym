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
NUM_GENERATIONS = 70  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500

#SCENARIO = 'Walker-v0'
SCENARIO = 'BridgeWalker-v0'

VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

CONTROLLER = alternating_gait
#CONTROLLER = sinusoidal_wave
#CONTROLLER = hopping_motion


# ---- INITIALIZATION ----
def create_random_robot():
    """Generate a valid random robot structure."""   
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot


# ---- MUTATION FUNCTION ----
def mutate_robot(parent):
    """Mutate the structure by changing a random voxel type."""
    child = copy.deepcopy(parent)
    x, y = np.random.randint(0, child.shape[0]), np.random.randint(0, child.shape[1])
    new_voxel = random.choice([v for v in VOXEL_TYPES if v != child[x, y]])  # Ensure mutation occurs by choosing a random voxel that isn't (x,y) 
    child[x, y] = new_voxel
    return child if is_connected(child) else parent  # If the mutation results in a disconnected robot, the function reverts back to the parent


# ---- (μ + λ) - EVOLUTION STRATEGIES  ---- 
def evolution_strategy():
    """Perform (μ + λ) - Evolution Strategies optimization and track fitness."""
    population = [create_random_robot() for _ in range(MU)]
    fitness_scores = [evaluate_fitness(robot) for robot in population]
    
    best_fitness_over_time = []  # Track best fitness per generation for plotting
    #avg_fitness_over_time = []

    for gen in range(NUM_GENERATIONS):
        offspring = [mutate_robot(random.choice(population)) for _ in range(LAMBDA)]
        offspring_fitness = [evaluate_fitness(robot) for robot in offspring]

        combined_population = population + offspring
        combined_fitness = fitness_scores + offspring_fitness
        sorted_indices = np.argsort(combined_fitness)[::-1]  

        population = [combined_population[i] for i in sorted_indices[:MU]]
        fitness_scores = [combined_fitness[i] for i in sorted_indices[:MU]]

        best_fitness = max(fitness_scores)
        #avg_fitness = np.mean(fitness_scores)

        best_fitness_over_time.append(best_fitness)
        #avg_fitness_over_time.append(avg_fitness)

        print(f"Generation {gen+1}: Best Fitness = {best_fitness}")

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

    return population[np.argmax(fitness_scores)], max(fitness_scores)  # The same as return best_robot, best_fitness because it's elitist


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
            #print(f"Step: {t}, Terminated: {terminated}, Truncated: {truncated}")
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

        # FITNESS CALCULATION
        # Speed bonus
        speed_score = t_reward / STEPS  

        # Actuator bonus (interval of numbers that make sense for this robot)
        actuator_count = np.count_nonzero(robot_structure == 4)
        if 2 <= actuator_count <= 4:
            actuator_bonus = 20  
        else:
            actuator_bonus = -10 * abs(actuator_count - 4)  

        # Finnishing simulation bonus
        if successful:
            t_reward *= 1.5  

        # Stability penalty (chaotic movement)
        reward_variance = np.var(reward_list)
        if reward_variance > 4:
            stability_penalty = -10
        else:
            stability_penalty = 0  

        # Final fitness score
        final_fitness = speed_score * 100 + actuator_bonus + stability_penalty
        return max(final_fitness, 0)  

    except (ValueError, IndexError):
        return 0.0


# (λ = 2μ) or (λ = 3μ) or (λ = 7μ)
MU = 5  
LAMBDA = 10  
best_robot, best_fitness = evolution_strategy()
print("Best robot structure found:")
print(best_robot)
print("Best fitness score:", best_fitness)

i = 0
while i < 5:
    utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
    i += 1
utils.create_gif(best_robot, filename='task1_bridge/ES_2/gif/ES2_70gen_500steps_4vrnc_4act.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)