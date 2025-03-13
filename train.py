import math
import random
import importlib.util
import numpy as np
import pickle
from simple_custom_taxi_env import SimpleTaxiEnv


def generate_random_map():
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)
    coor = [(i, j) for i in range(env.grid_size) for j in range(env.grid_size)]
    random.shuffle(coor)
    env.stations = coor[:4]
    # env.obstacles = coor[4:8]
    env.obstacles = []
    return env


def train(episodes=5000, alpha=0.1, gamma=0.99):
    spec = importlib.util.spec_from_file_location("student_agent", "student_agent.py")
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    policy_table = {}
    rewards_per_episode = []

    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    for episode in range(episodes):
        # print(f'=== Episode {episode + 1} ===')
        env = generate_random_map()
        obs, _ = env.reset()
        state, _ = student_agent.my_state(obs)
        done = False
        total_reward = 0
        trajectory = []

        # Reset Global Variable in student_agent
        student_agent.is_picked = False
        student_agent.visited = {"R": 0, "G": 0, "Y": 0, "B": 0}
        student_agent.destination = None

        while not done:
            if state == 4 or state == 5:
                action = state

            else:
                if state not in policy_table.keys():
                    policy_table[state] = np.zeros(4)

                action_probs = softmax(policy_table[state])
                action = np.random.choice([0, 1, 2, 3], p=action_probs)

            obs, _, done, _ = env.step(action)
            next_state, reward = student_agent.my_state(obs)
            total_reward += reward

            if state not in [4, 5]:
                trajectory.append((state, action, reward))

            if reward == 1:
                # Test Only
                # print('--------')
                # for t in trajectory:
                    # print(t[0], env.get_action_name(t[1]), t[2])

                # del trajectory[0]
                G = 0  # Return (discounted sum of rewards)
                for t in reversed(range(len(trajectory))):
                    s, a, r = trajectory[t]
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
    with open('policy.pkl', 'wb') as f:
        pickle.dump(policy_table, f)
