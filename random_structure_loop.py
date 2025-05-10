import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from controllers_fixed import *

NUM_GENERATIONS = 800 #walker
# NUM_GENERATIONS = 1000 #bridge
NUM_LOOPS = 5
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
STEPS = 500

#SCENARIO = 'BridgeWalker-v0'
SCENARIO = 'Walker-v0'

VOXEL_TYPES = [0, 1, 2, 3, 4]

CONTROLLER = alternating_gait

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
        start_pos = np.mean(sim.object_pos_at_time(0, "robot")[0])
        for t in range(STEPS):  
            # Update actuation before stepping
            actuation = CONTROLLER(action_size,t)
            if view:
                viewer.render('screen')  #modo view para por no ecra, podemos nao fazer
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward

            if terminated or truncated:
                break

        end_pos = np.mean(sim.object_pos_at_time(sim.get_time(), "robot")[0])
        distance_traveled = end_pos - start_pos
        viewer.close()
        env.close()
        return t_reward, distance_traveled
    except (ValueError, IndexError) as e:
        return 0.0

def create_random_robot():
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot

def run_multiple_random_searches():
    for loop in range(NUM_LOOPS):
        print(f"\n--- Starting Random Search Loop {loop + 1} ---")
        best_robot = None
        best_fitness = -float('inf')
        best_distance = -float('inf')
        
        for gen in range(NUM_GENERATIONS):
            robot = create_random_robot()
            fitness_score = evaluate_fitness(robot)
            
            if fitness_score[0] > best_fitness:
                best_fitness = fitness_score[0]
                best_robot = robot
                best_distance = fitness_score[1]

            print(f"[Loop {loop+1}] Generation {gen+1}: Fitness = {fitness_score}, Distance = {best_distance}")

        # Save GIF of the best robot for this loop
        gif_filename = f'task1_walker/Random_No_fitness/Random_No_fitness_{loop+1}.gif'
        #gif_filename = f'task1_bridge/new_Random/New_Random_{loop+1}.gif'
        utils.create_gif(best_robot, filename=gif_filename, scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)

        # Save structure and fitness to .txt
        txt_filename = f'task1_walker/Random_No_fitness/Random_No_fitness_{loop+1}.txt'
        #txt_filename = f'task1_walker/new_Random/Random_{loop+1}.txt'
        with open(txt_filename, 'w') as f:
            f.write(f"RUN {loop+1}\n")
            f.write(f"Best fitness: {best_fitness}\n")
            f.write(f"Best distance: {best_distance}\n")
            f.write("Best robot structure:\n")
            f.write(str(best_robot))

        print(f"Best fitness in Loop {loop+1}: {best_fitness}, Distance: {best_distance}")

run_multiple_random_searches()