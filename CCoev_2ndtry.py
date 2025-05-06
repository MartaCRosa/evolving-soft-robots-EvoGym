import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
from evogym import get_full_connectivity
import matplotlib.pyplot as plt
import torch
from controller_neural2 import NeuralController, get_weights, set_weights

# ---- PARAMETERS ----
NUM_GENERATIONS = 5
POP_SIZE = 6
TOURNAMENT_SIZE = 3
STEPS = 500
SEED = 40
np.random.seed(SEED)
random.seed(SEED)

SCENARIO = 'GapJumper-v0'

def generate_fixed_shape():
    return np.random.choice([0, 1, 2, 3, 4], size=(5, 5))

class CoopCoevolutionPaired:
    def __init__(self, input_size, output_size, pop_size=POP_SIZE, tournament_size=TOURNAMENT_SIZE, fixed_shape=None):
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.input_size = input_size
        self.output_size = output_size
        self.fixed_shape = fixed_shape

        # Paired population: (structure, controller_weights)
        self.population = [(self.create_random_robot(), self.random_weights()) for _ in range(self.pop_size)]

        self.best_fitness_per_generation = []

    def create_random_robot(self):
        if self.fixed_shape is not None:
            return self.fixed_shape.copy()
        return np.random.choice([0, 1, 2, 3, 4], size=(5, 5))

    def random_weights(self):
        controller = NeuralController(self.input_size, self.output_size)
        return get_weights(controller)

    def set_weights_into_controller(self, weights):
        model = NeuralController(self.input_size, self.output_size)
        set_weights(model, weights)
        return model.eval()

    def evaluate(self, structure, weights):
        # Count number of motor voxels (3 and 4)
        num_motors = int(np.sum(np.isin(structure, [3, 4])))

        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=structure, connections=get_full_connectivity(structure))
        state, _ = env.reset()
        total_reward = 0

        # Build controller with dynamic output size
        controller = NeuralController(self.input_size, num_motors)
        set_weights(controller, weights)
        controller.eval()

        for _ in range(STEPS):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = controller(state_tensor).detach().numpy().flatten()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        env.close()
        return total_reward, controller


    def struct_crossover(self, r1, r2):
        min_shape = tuple(np.minimum(r1.shape, r2.shape))
        mask = np.random.rand(*min_shape) > 0.5
        child = np.zeros_like(r1)
        child[:min_shape[0], :min_shape[1]] = np.where(mask, r1[:min_shape[0], :min_shape[1]], r2[:min_shape[0], :min_shape[1]])
        return child

    def struct_mutation(self, structure, mutation_rate=0.1):
        mutated = structure.copy()
        for i in range(mutated.shape[0]):
            for j in range(mutated.shape[1]):
                if random.random() < mutation_rate:
                    mutated[i, j] = random.choice([0, 1, 2, 3, 4])
        return mutated

    def weights_crossover(self, w1, w2):
        return [(alpha := random.random()) * a + (1 - alpha) * b for a, b in zip(w1, w2)]

    def weights_mutation(self, weights, mutation_rate=0.1):
        return [w + mutation_rate * np.random.randn(*w.shape) for w in weights]

    def tournament_selection(self, population, fitnesses):
        winners = []
        for _ in range(self.pop_size):
            participants = random.sample(list(zip(population, fitnesses)), self.tournament_size)
            winner = max(participants, key=lambda x: x[1])[0]
            winners.append(winner)
        return winners

    def next_generation(self, fitnesses):
        selected = self.tournament_selection(self.population, fitnesses)
        next_gen = []

        for _ in range(self.pop_size // 2):
            (s1, w1), (s2, w2) = random.sample(selected, 2)
            child_struct1 = self.struct_mutation(self.struct_crossover(s1, s2))
            child_struct2 = self.struct_mutation(self.struct_crossover(s2, s1))
            child_weights1 = self.weights_mutation(self.weights_crossover(w1, w2))
            child_weights2 = self.weights_mutation(self.weights_crossover(w2, w1))
            next_gen.append((child_struct1, child_weights1))
            next_gen.append((child_struct2, child_weights2))

        # Elitism
        best_idx = np.argmax(fitnesses)
        best = copy.deepcopy(self.population[best_idx])
        next_gen[0] = best

        return next_gen

    def plot_fitness(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.best_fitness_per_generation, marker='o')
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Fitness Over Generations")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def train(self, generations=NUM_GENERATIONS):
        for gen in range(generations):
            fitnesses = [self.evaluate(struct, weights)[0] for struct, weights in self.population]
            best_idx = np.argmax(fitnesses)
            best_fit = fitnesses[best_idx]
            self.best_fitness_per_generation.append(best_fit)
            print(f"Generation {gen}: Best Fitness = {best_fit:.2f}")
            self.population = self.next_generation(fitnesses)

        self.plot_fitness()

# ---------- MAIN ----------

robot_structure = generate_fixed_shape()
connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)

env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
print(f"Action space shape: {env.action_space.shape}")
print(f"Action space: {env.action_space}")
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Observation space: {env.observation_space}")

input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
env.close()

evo = CoopCoevolutionPaired(input_size, output_size, fixed_shape=robot_structure)
evo.train()
