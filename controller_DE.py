import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from controller_neural import *
import matplotlib.pyplot as plt

# --- EvoGym Setup ---
NUM_GENERATIONS = 45  # Number of generations to evolve
STEPS = 500
SEED = 42  # Set random seed for reproducibility
np.random.seed(SEED)
random.seed(SEED)

SCENARIO = 'DownStepper-v0'
#SCENARIO = 'ObstacleTraverser-v0'

robot_structure = np.array([ 
[1,3,1,0,0],
[4,1,3,2,2],
[3,4,4,4,4],
[3,0,0,3,2],
[0,0,0,0,2]
])

connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]  # Observation size
output_size = env.action_space.shape[0]  # Action size


# ---- CONTROLLER ----
brain = NeuralController(input_size, output_size)

# ---- WEIGHT UTILITIES ----
def flatten_weights(weights): #para podemros trabalhar com os pesos da rede 
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

# --- Fitness Function ---
def evaluate_fitness(weights, view=False):
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    if view:
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

    state = env.reset()[0]
    t_reward = 0
    velocity_list = []
    start_pos = np.mean(sim.object_pos_at_time(0, "robot")[0])
    active_time = 0

    for _ in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        if view:
            viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        t_reward += reward

        velocities = sim.vel_at_time(sim.get_time())
        avg_x_velocity = np.mean(velocities[0])
        velocity_list.append(avg_x_velocity)

        active_time += 1

        if terminated or truncated:
            break

    end_pos = np.mean(sim.object_pos_at_time(sim.get_time(), "robot")[0])
    distance_traveled = end_pos - start_pos
    avg_velocity = np.mean(velocity_list)

    # --- Fitness calculation ---
    distance_bonus = distance_traveled * 20 if distance_traveled > 0 else distance_traveled * 50
    velocity_bonus = avg_velocity * 50 if avg_velocity > 0 else avg_velocity * 50
    fall_penalty = -200 if terminated and not truncated else 0

    final_fitness = t_reward + distance_bonus + velocity_bonus + fall_penalty

    if view:
        viewer.close()
    env.close()

    print(f"Distance: {distance_traveled:.4f}, Velocity: {avg_velocity:.4f}, Time: {active_time}, Final: {final_fitness:.4f}")

    return final_fitness, distance_traveled


POP_SIZE = 50
MAX_GAMMA = 1.0
MIN_GAMMA = 0.3
CROSSV_RATE = 0.7
BOUNDS = (-2,2)

initial_weights = flatten_weights(get_weights(brain))
dim = len(initial_weights)

population = [np.random.uniform(BOUNDS[0], BOUNDS[1], dim) for _ in range(POP_SIZE)]


""" DE notion --> DE/rand/1/bin: """
# - rand - strategie of base vetor selection (x1): random selection of base vector for mutation
# - 1 - difference vector(s): just 1 difference vector (x2-x3)
# - bin - crossover operator: crossover binomial
# changing rand to best increases selective pressure but decreases diversity
# increading number of difference vectors increases diversity

gamma = 0 #scale factor 
#--> increasing the scale factor: promotes exploration (bigger steps, more random)
#--> decreasing the scale factor: promotes exploitation (finner search near current good solutions)
# the goal will be to not keep this factor the same thorugh generations. starting with it bigger
# and decreasing it through generations! this addaption can also be random. we start with linear first

""" Mutation """
# --> A mutant vector is created using 3 individuals from the population. These 3 individuals are not changed (acho eu)
# The scaling factor influences the exploration and exploitation of mutation


""" Crossover """
# --> Crossover happens between a target individual and the mutant vector
# for selection of survivor of the next generation. The result is an individual called trial

""" Selection for survival of next generation """
# --> Trial and target individuals are compared and whoever has the best fitness is kept for the next generation.
# As one trial individual is created for every selected target individual but only one of the 2 is kept, the next generation 
# keeps the same number of individuals as originally


# ---- ADAPTATIVE LINEAR SCALE FUNCTION -----
def adaptative_linear_scale_factor(max_gamma, min_gamma, num_iter, current_iter):
    return max_gamma - ((max_gamma - min_gamma) * (current_iter / num_iter))


# ---- ADAPTATIVE RANDOM SCALE FUNCTION -----
def adpatative_random_scale_factor():
    return 0.5*(1 - np.random.rand())

# ---- MUTATION ------
def mutation(pop, variant_idx):
    # variant_idx is the index of the individual to be mutated
    # defining the index of the 3 randomly choosen individuals
    idxs = [i for i in range(POP_SIZE) if i != variant_idx] # ensure the individual mutated is not used to mutate itself
    a, b, c = random.sample(idxs,3) # gets 3 random individuals 
    scale_factor = adaptative_linear_scale_factor(MAX_GAMMA, MIN_GAMMA, NUM_GENERATIONS, gen) #alterar para random para testar
    mutant_vector = pop[a] - scale_factor*(pop[b] - pop[c]) # from formula variant = base + scale*difference (base-x1; difference x2-x3)
    return np.clip(mutant_vector, BOUNDS[0], BOUNDS[1])

# ---- BINOMIAL CROSSOVER -----
def crossover(target, variant):
    trial = np.copy(target)
    for i in range(len(target)):
        if random.random() < CROSSV_RATE or i == random.randint(0, dim - 1):
            trial[i] = variant[i] #when this condition is not met the part of the trial[i] remains equal to the target[i]
    return trial

# ----- DE LOOP ------
initial_weights = flatten_weights(get_weights(brain))
dim = len(initial_weights)

population = [np.random.uniform(BOUNDS[0], BOUNDS[1], dim) for _ in range(POP_SIZE)]

fitness_history = []
avg_fitness_history = []
best_distance_history = []

# Pre-evaluate the initial population
fitness_cache = []
distance_cache = []
for individual in population:
    weights = reshape_weights(individual, brain)
    fitness, distance = evaluate_fitness(weights)
    fitness_cache.append(fitness)
    distance_cache.append(distance)

for gen in range(NUM_GENERATIONS):
    new_population = []
    new_fitness_cache = []
    new_distance_cache = []

    for i in range(POP_SIZE):
        target = population[i]
        target_fitness = fitness_cache[i]
        target_distance = distance_cache[i]

        mutant_vector = mutation(population, i)
        trial = crossover(target, mutant_vector)
        trial_weights = reshape_weights(trial, brain)
        trial_fitness, trial_distance = evaluate_fitness(trial_weights)

        # Selection
        if trial_fitness > target_fitness:
            new_population.append(trial)
            new_fitness_cache.append(trial_fitness)
            new_distance_cache.append(trial_distance)
        else:
            new_population.append(target)
            new_fitness_cache.append(target_fitness)
            new_distance_cache.append(target_distance)

    population = new_population
    fitness_cache = new_fitness_cache
    distance_cache = new_distance_cache

    best_idx = np.argmax(fitness_cache)
    best_fitness = fitness_cache[best_idx]
    best_distance = distance_cache[best_idx]
    avg_fitness = np.mean(fitness_cache)

    print(f"[Gen {gen+1}] Best Fitness: {best_fitness:.6f} | Distance: {best_distance:.6f}")
    fitness_history.append(best_fitness)
    avg_fitness_history.append(avg_fitness)
    best_distance_history.append(best_distance)
    

# --- Final best weights ---
best_weights = reshape_weights(population[best_idx], brain)
set_weights(brain, best_weights)

# --- Fitness Plot ---
plt.figure(figsize=(8, 5))
plt.plot(range(NUM_GENERATIONS), fitness_history, label="Best Fitness", color="blue")
plt.plot(range(NUM_GENERATIONS), avg_fitness_history, label="Average Fitness", color="orange", linestyle="dashed")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("DE Fitness Progression")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# --- Visualization of Best Controller ---
def visualize_policy(weights):
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    viewer = EvoViewer(env.sim)
    viewer.track_objects('robot')
    state = env.reset()[0]
    for _ in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    viewer.close()
    env.close()

# Visualize multiple times
for _ in range(5):
    visualize_policy(best_weights)
