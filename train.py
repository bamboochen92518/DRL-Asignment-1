import math
import random
import importlib.util
import numpy as np
import json
from simple_custom_taxi_env import SimpleTaxiEnv
from save_policy_table import save_policy_table


DEBUG = False
SWAP_TIME = 200
OBSTACLE_CNT = 3


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def check_stations(stations):
    for i in range(0, 4):
        for j in range(i + 1, 4):
            if distance(stations[i], stations[j]) == 1:
                return False
    return True


def generate_random_map():
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)
    coor = [(i, j) for i in range(env.grid_size) for j in range(env.grid_size)]
    success = False
    while not success:
        random.shuffle(coor)
        env.stations = coor[:4]
        success = check_stations(env.stations)
        # env.stations = [(0, 0), (0, 4), (4, 0), (4, 4)]
    obstacle = random.randint(4, 4 + OBSTACLE_CNT)
    env.obstacles = coor[4:obstacle]
    return env


def train(episodes=36888, alpha=0.1, gamma=0.99):
    spec = importlib.util.spec_from_file_location("student_agent", "student_agent.py")
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    policy_table = {}
    rewards_per_episode = []

    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    for episode in range(episodes):
        if episode % SWAP_TIME == 0:
            env = generate_random_map()

        obs, _ = env.reset()

        if DEBUG:
            print(f'=== Episode {episode + 1} ===')
            print(f'GLOBAL: stop = {env.stations}')

        state, _, o1, o2, o3, o4 = student_agent.my_state(obs)
        state = (state, o1, o2, o3, o4)
        done = False
        total_reward = 0
        trajectory = []

        # Reset Global Variable in student_agent
        student_agent.is_picked = False
        student_agent.visited = {"R": 0, "G": 0, "Y": 0, "B": 0}
        student_agent.destination = None

        while not done:
            if isinstance(state[0], int):
                action = state[0]

            else:
                if state not in policy_table.keys():
                    policy_table[state] = np.zeros(4)

                action_probs = softmax(policy_table[state])
                action = np.random.choice([0, 1, 2, 3], p=action_probs)

            obs, sys_reward, done, _ = env.step(action)
            next_state, reward, no1, no2, no3, no4 = student_agent.my_state(obs)
            next_state = (next_state, no1, no2, no3, no4)

            if sys_reward == -5:
                reward = -1

            if not isinstance(state[0], int):
                trajectory.append((state, action, reward))

            if reward == 1:
                G = 0  # Return (discounted sum of rewards)
                for t in reversed(range(len(trajectory))):
                    s, a, r = trajectory[t]
                    if t == len(trajectory) - 1:
                        r = 1 - 0.9 * ((t + 1) / 5000)
                    total_reward += r
                    G = r + gamma * G  # Discounted reward

                    pi = softmax(policy_table[s])
                    grad_log_pi = -pi
                    grad_log_pi[a] += 1
                    policy_table[s] += alpha * G * grad_log_pi

                rewards_per_episode.append(total_reward)
                total_reward = 0
                trajectory = []

            state = next_state

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode)
            rewards_per_episode = []
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}")

    return policy_table, rewards_per_episode


if __name__ == "__main__":
    policy_table, rewards = train()
    save_policy_table(policy_table)
