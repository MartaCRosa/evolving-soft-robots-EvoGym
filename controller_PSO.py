import gym
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from evogym import get_full_connectivity
from evogym.viewer import EvoViewer

# --- EvoGym Setup ---
NUM_ITERATIONS = 10
STEPS = 500
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

SCENARIO = 'DownStepper-v0'

robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])

connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

# ---- CONTROLLER ----
class NeuralController(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

brain = NeuralController(input_size, output_size)

def get_weights(model):
    return [p.data.numpy() for p in model.parameters()]

def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data = torch.tensor(w, dtype=torch.float32)

def flatten_weights(weights):
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

        if terminated or truncated:
            break

    end_pos = np.mean(sim.object_pos_at_time(sim.get_time(), "robot")[0])
    distance_traveled = end_pos - start_pos

    distance_bonus = distance_traveled * 2 if distance_traveled > 0 else -2
    avg_velocity = np.mean(velocity_list)
    velocity_bonus = avg_velocity * 3 if avg_velocity > 0 else -2
    fall_penalty = -2 if terminated and not truncated else 0

    final_fitness = t_reward + distance_bonus + velocity_bonus + fall_penalty
    print(f"Distance: {distance_traveled:.4f}, Velocity: {avg_velocity:.4f}, Fall: {fall_penalty}, Final: {final_fitness:.4f}")

    if view:
        viewer.close()
    env.close()
    return final_fitness

# --- PSO Setup ---
NUM_PARTICLES = 20
W = 0.5
C1 = 1.5
C2 = 1.5

initial_weights = get_weights(brain)
flat_len = len(flatten_weights(initial_weights))
bounds = np.array([(-2, 2)] * flat_len)

swarm_pos = np.random.uniform(low=-2, high=2, size=(NUM_PARTICLES, flat_len))
swarm_vel = np.random.uniform(low=-0.1, high=0.1, size=(NUM_PARTICLES, flat_len))
personal_best_pos = np.copy(swarm_pos)
personal_best_scores = np.array([evaluate_fitness(reshape_weights(p, brain)) for p in personal_best_pos])
global_best_idx = np.argmax(personal_best_scores)
global_best_pos = personal_best_pos[global_best_idx]
global_best_score = personal_best_scores[global_best_idx]

fitness_progress = []

for iteration in range(NUM_ITERATIONS):
    for i in range(NUM_PARTICLES):
        r1 = np.random.rand(flat_len)
        r2 = np.random.rand(flat_len)

        swarm_vel[i] = (
            W * swarm_vel[i] +
            C1 * r1 * (personal_best_pos[i] - swarm_pos[i]) +
            C2 * r2 * (global_best_pos - swarm_pos[i])
        )
        swarm_pos[i] += swarm_vel[i]
        swarm_pos[i] = np.clip(swarm_pos[i], bounds[:, 0], bounds[:, 1])

        fitness = evaluate_fitness(reshape_weights(swarm_pos[i], brain))
        if fitness > personal_best_scores[i]:
            personal_best_scores[i] = fitness
            personal_best_pos[i] = swarm_pos[i]

            if fitness > global_best_score:
                global_best_score = fitness
                global_best_pos = swarm_pos[i]

    fitness_progress.append(global_best_score)
    print(f"Generation {iteration+1}/{NUM_ITERATIONS}, Best Fitness: {global_best_score:.5f}")

# Set best weights
set_weights(brain, reshape_weights(global_best_pos, brain))

# Plot
plt.plot(fitness_progress)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("PSO Fitness Progress")
plt.grid(True)
plt.show()

# --- Visualize Best Policy ---
def visualize_policy(weights):
    set_weights(brain, weights)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    viewer = EvoViewer(env.sim)
    viewer.track_objects('robot')
    state = env.reset()[0]
    for t in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    viewer.close()
    env.close()

# Run multiple times to see the behavior
i = 0
while i < 5:
    visualize_policy(reshape_weights(global_best_pos, brain))
    i += 1
