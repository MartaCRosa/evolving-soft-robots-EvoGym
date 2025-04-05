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
NUM_GENERATIONS = 100  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500

SCENARIO = 'Walker-v0'
#SCENARIO = 'BridgeWalker-v0'

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
    """Perform (μ + λ) - ES optimization and track fitnesses."""
    population = [create_random_robot() for _ in range(MU)]  # Create mu random robots
    fitness_scores = [evaluate_fitness(robot) for robot in population]  # Evaluate
    
    best_fitness_over_time = []  # Track best fitness per generation for plotting
    avg_fitness_over_time = []

    for gen in range(NUM_GENERATIONS):
        offspring = [mutate_robot(random.choice(population)) for _ in range(LAMBDA)]  # Generate lambda offspring with mutation
        offspring_fitness = [evaluate_fitness(robot) for robot in offspring]  # Evaluate offspring

        combined_population = population + offspring
        combined_fitness = fitness_scores + offspring_fitness
        sorted_indices = np.argsort(combined_fitness)[::-1]  # Sort by fitness the combination of mu + lambda

        population = [combined_population[i] for i in sorted_indices[:MU]]  # Choose mu best to become parents of next population
        fitness_scores = [combined_fitness[i] for i in sorted_indices[:MU]]

        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)

        best_fitness_over_time.append(best_fitness)
        avg_fitness_over_time.append(avg_fitness)

        print(f"Generation {gen+1}: Best Fitness = {best_fitness}")

    # Plot fitness vs. generations
    plt.figure(figsize=(8, 5))
    plt.plot(range(NUM_GENERATIONS), best_fitness_over_time, label="Best Fitness", color="blue")
    plt.plot(range(NUM_GENERATIONS), avg_fitness_over_time, label="Average Fitness", color="orange", linestyle="dashed")
    
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("ES: Fitness Progression")
    plt.legend()
    plt.grid()
    plt.show()

    return population[np.argmax(fitness_scores)], max(fitness_scores)  # The same as return best_robot, best_fitness because it's elitist


# ---- EVALUATION FUNCTION ----
def evaluate_fitness(robot_structure, view=False): 
    """Evaluate the fitness by taking into account: evogym reward, distance travelled, velocity and actuator number."""   
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
        #reward_list = []  # Store rewards per step
        velocity_list = []  # Store velocity per step        
        start_pos = np.mean(sim.object_pos_at_time(0, "robot")[0]) # Get initial position (center of mass)

        for t in range(STEPS):  
            actuation = CONTROLLER(action_size, t)
            if view:
                viewer.render('screen')
            ob, reward, terminated, truncated, info = env.step(actuation)  
            #print(f"Step: {t}, Terminated: {terminated}, Truncated: {truncated}")
            t_reward += reward  
            #reward_list.append(reward)

            # Get velocity at this timestep
            velocities = sim.vel_at_time(sim.get_time())  # (2, n) array (x, y velocities)
            avg_x_velocity = np.mean(velocities[0])  # Take mean x-velocity
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
            t_reward *= 1.5  

        # Distance traveled bonus
        distance_traveled = end_pos - start_pos  # Positive means moving forward
        distance_bonus = max(distance_traveled * 1.3, 0)

        # Velocity bonus
        avg_velocity = np.mean(velocity_list)  
        velocity_bonus = max(avg_velocity * 1.3, 0)

        # Actuator(-) bonus (interval of numbers that make sense for the environment)
        actuator_count = np.count_nonzero(robot_structure == 4)
        if 2 <= actuator_count <= 4:    #2-4 walker, 3-7 bridge
            actuator_bonus = 10  
        else:
            actuator_bonus = -5  

        """
        # Stability penalty (High variance == chaotic movement with sudden reward spikes and drops)
        reward_variance = np.var(reward_list)
        print(f"Reward Variance: {reward_variance}")
        if reward_variance > 1:
            stability_penalty = -10
        else:
            stability_penalty = 0  
        
        # Leg Bonus (ensure legs at the sides and a gap in the middle)
        bottom_rows = robot_structure[3:, :]
        leg_bonus = 0

        if bottom_rows[0, 0] != 0 and bottom_rows[1, 0] != 0:  # Check both rows for left leg in column 0
            leg_bonus += 3
        if bottom_rows[0, 1] != 0 and bottom_rows[1, 1] != 0:  # Check both rows for left leg in column 1
            leg_bonus += 1.5
        if bottom_rows[0, 3] != 0 and bottom_rows[1, 3] != 0:  # Check both rows for right leg in column 3
            leg_bonus += 3
        if bottom_rows[0, 4] != 0 and bottom_rows[1, 4] != 0:  # Check both rows for right leg in column 4
            leg_bonus += 1.5

        if bottom_rows[0, 2] != 0:  # If center leg is present in any row
            leg_bonus -= 3
        if bottom_rows[1, 2] != 0:
            leg_bonus-=1.5
        """

        # Final fitness score
        final_fitness = t_reward + distance_bonus + velocity_bonus + actuator_bonus #+ stability_penalty + leg_bonus
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
utils.create_gif(best_robot, filename='task1_walker/ES/ES_100gen_5.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)