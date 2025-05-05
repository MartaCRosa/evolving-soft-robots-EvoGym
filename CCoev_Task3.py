import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from controller_neural import *
import matplotlib.pyplot as plt
import torch

# ---- PARAMETERS ----
NUM_GENERATIONS = 5
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
STEPS = 500
SEED = 40
np.random.seed(SEED)
random.seed(SEED)

SCENARIO = 'GapJumper-v0'

def generate_fixed_shape():
    return np.random.choice([0, 1, 2, 3, 4], size=(5, 5))

class CoopCoevolution:
    def __init__(self, input_size, output_size, pop_size=6, tournament_size=3, fixed_shape=None):
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.input_size = input_size
        self.output_size = output_size

        self.fixed_shape = fixed_shape

        self.pop_struct = [self.create_random_robot() for _ in range(pop_size)]
        self.pop_contr = [self.random_weights() for _ in range(pop_size)]

        self.best_fitness_per_generation = []

    def create_random_robot(self):
        # Use fixed shape if provided
        if self.fixed_shape is not None:
            shape = self.fixed_shape.shape
        else:
            shape = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
        return np.random.choice([0, 1, 2, 3, 4], size=shape)


    def random_weights(self):
        controller = NeuralController(self.input_size, self.output_size)
        return get_weights(controller)

    def set_weights_into_controller(self, weights):
        model = NeuralController(self.input_size, self.output_size)
        set_weights(model, weights)
        return model.eval()

    def evaluate(self, structure, weights):
        connectivity = get_full_connectivity(structure)
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=structure, connections=connectivity)
        state = env.reset(seed=SEED)[0]

        controller = self.set_weights_into_controller(weights)
        t_reward = 0
        velocity_list = []
        start_pos = np.mean(env.sim.object_pos_at_time(0, "robot")[0])

        for _ in range(STEPS):

            # checkar primeiro o tamanho do tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Check and adjust input size if necessary
            if state_tensor.shape[1] != self.input_size:
                print(f"[WARNING] Input size mismatch: expected {self.input_size}, got {state_tensor.shape[1]}. Adjusting.")
                self.input_size = state_tensor.shape[1]
                controller = self.set_weights_into_controller(weights)

            action = controller(state_tensor).detach().numpy().flatten()
            state, reward, terminated, truncated, _ = env.step(action)
            t_reward += reward

            velocities = env.sim.vel_at_time(env.sim.get_time())
            avg_x_velocity = np.mean(velocities[0])
            velocity_list.append(avg_x_velocity)
            
            if terminated or truncated:
                break

        end_pos = np.mean(env.sim.object_pos_at_time(env.sim.get_time(), "robot")[0])
        distance_traveled = end_pos - start_pos
        avg_velocity = np.mean(velocity_list)

        distance_bonus = distance_traveled * 20 if distance_traveled > 0 else distance_traveled * 50
        velocity_bonus = avg_velocity * 50
        fall_penalty = -200 if terminated and not truncated else 0

        final_fitness = t_reward + distance_bonus + velocity_bonus + fall_penalty
        env.close()
        return final_fitness

    def tournament_selection(self, population, fitnesses):
        winners = []
        for _ in range(self.pop_size):
            participants = random.sample(list(zip(population, fitnesses)), self.tournament_size)
            winner = max(participants, key=lambda x: x[1])[0]
            winners.append(winner)
        return winners

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
            best_pair, best_fitness = self.random_paring_co_evolution()
            self.best_fitness_per_generation.append(best_fitness)
            print(f"Generation {gen}: Best Fitness = {best_fitness:.2f}")
        self.plot_fitness()

    def select_and_generate_next(self, population, fitnesses, is_weights=False):
        selected = self.tournament_selection(population, fitnesses)
        next_gen = []
        for _ in range(self.pop_size // 2):
            p1, p2 = random.sample(selected, 2)
            if is_weights:
                c1 = self.weights_mutation(self.weights_crossover(p1, p2))
                c2 = self.weights_mutation(self.weights_crossover(p2, p1))
            else:
                c1 = self.struct_mutation(self.struct_crossover(p1, p2))
                c2 = self.struct_mutation(self.struct_crossover(p2, p1))
            next_gen.extend([c1, c2])
        return next_gen

    def random_paring_co_evolution(self):
            fitness_structures = []
            fitness_controllers = []
            pairings = []

            """ Notas """
            # Nesta task a estrutura e o controlador já não são soluções individuais e por isso o seu fitness não pode ser avaliado de forma individual.
            # O que acontece é que as estruturas cooperam com os controladores (e vice-versa) para evoluir os robots
            # Assim o fitness da estrutura e do controlador é avaliado para cada um segundo a forma como cada um deste coopera com o outro.
            # Se uma estrtura coopera bem com um controlador selecionado de forma random o fitness desta estrutura é bom e não simplesmente tem um 
            # fitness bom porque é uma boa estrtura. A mesma lógica para o controlador.
            
            # Evaluation of structures 
            for structure in self.pop_struct:
                partner = random.choice(self.pop_contr) # controller is choosen randomly
                fit, _ = self.evaluate(structure, partner)  # paring is done between the structure being evaluated and the controller randomly choosen. fitness of this pairing is evaluated
                fitness_structures.append(fit) 
                pairings.append((structure, partner, fit)) # to keep info of the parings that resulted in the fitnesses for the structures

            # Evaluation of controllers 
            for controller in self.pop_contr:
                partner = random.choice(self.pop_struct) # structure is choosen randomly
                fit, _ = self.evaluate(partner, controller) # paring is done between the structure being evaluated and the controller randomly choosen. fitness of this pairing is evaluated
                fitness_controllers.append(fit)
                pairings.append((partner, controller, fit)) # to keep info of the parings that resulted in the fitnesses for the controllers

                # -> pairing is going to be a list ot tupples, where each tuple is:
                # (structure, controller, fitness) fitness resulting from the pairing of structure and controller

            # check best solution
            best_struct, best_contr, best_fit = max(pairings, key=lambda t: t[2]) # gets best fitness and gives the correspondent structure and controller that originates that fitness
        
            # selection, variation operators, offspring
            self.pop_struct = self.select_and_generate_next(self.pop_struct, fitness_structures)
            self.pop_contr = self.select_and_generate_next(self.pop_contr, fitness_controllers)
            
            # elitism pop_struct
            self.pop_struct[0] = best_struct.copy() # the first individual os structures is the best from the pairings -> elitism
            
            # elitism pop_contr
            self.pop_contr[0] = best_contr.copy() # the same for the controllers population 
            
            return (best_struct.copy(), best_contr.copy()), best_fit
    

robot_structure = generate_fixed_shape()

# Now you can use this robot structure
connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
env.close()

# Initialize CoopCoevolution with the generated robot structure
evo = CoopCoevolution(input_size, output_size, fixed_shape=robot_structure)
evo.train()



