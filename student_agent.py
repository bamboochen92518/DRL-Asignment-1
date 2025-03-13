# Remember to adjust your student ID in meta.xml
import numpy as np
import json
import random
import gym


is_picked = False
visited = {"R": 0, "G": 0, "Y": 0, "B": 0}
destination = None


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numeric stability
    return exp_x / exp_x.sum()


def my_state(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    # Load global variable
    global is_picked
    global visited
    global destination

    taxi_row, taxi_col, Rx, Ry, Gx, Gy, Yx, Yy, Bx, By, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    destination_to_loc = {'R': (Rx, Ry), 'G': (Gx, Gy), 'Y': (Yx, Yy), 'B': (Bx, By)}

    # Determine where to go
    def determine_target():
        target = (0, 0)
        if not is_picked:
            if visited['R'] == 0:
                target = (Rx, Ry)
            elif visited['G'] == 0:
                target = (Gx, Gy)
            elif visited['Y'] == 0:
                target = (Yx, Yy)
            else:
                target = (Bx, By)
        else:
            if destination:
                target = destination_to_loc[destination]
            elif visited['R'] == 1:
                target = (Rx, Ry)
            elif visited['G'] == 1:
                target = (Gx, Gy)
            elif visited['Y'] == 1:
                target = (Yx, Yy)
            else:
                target = (Bx, By)
        return target
    
    # Determine the spot
    def on_location():
        taxi_loc = (taxi_row, taxi_col)
        if taxi_loc == (Rx, Ry):
            return 'R'
        if taxi_loc == (Gx, Gy):
            return 'G'
        if taxi_loc == (Yx, Yy):
            return 'Y'
        if taxi_loc == (Bx, By):
            return 'B'
        return None
    
    curr_target = determine_target()
    reward = -0.01
    direction = (taxi_row - curr_target[0], taxi_col - curr_target[1])
    
    # Not Arrive Yet
    if direction != (0, 0):
        return direction, reward
    
    # Arrive
    reward = 1

    # Passenger is not picked yet
    if not is_picked:
        if passenger_look:
            is_picked = True
            visited['R'] = 1
            visited['G'] = 1
            visited['Y'] = 1
            visited['B'] = 1
            visited[on_location()] += 1
            return 4, reward
        else:
            if destination_look:
                destination = on_location()
            visited[on_location()] = 1
            next_target = determine_target()
            direction = (taxi_row - next_target[0], taxi_col - next_target[1])
            return direction, reward

    # Passenger is picked
    if destination_look:
        return 5, reward

    visited[on_location()] += 1
    next_target = determine_target()
    direction = (taxi_row - next_target[0], taxi_col - next_target[1])
    return direction, reward


def get_action(obs):
    direction, _ = my_state(obs)
    if isinstance(direction, int):
        return direction

    while abs(direction[0]) > 4 or abs(direction[1]) > 4:
        direction = (direction[0] // 2, direction[1] // 2)
    
    direction = str(direction)

    with open('policy.json', 'r') as f:
        policy_table = json.load(f)
    
    action_prob = softmax(policy_table[direction])
    
    return np.random.choice(len(action_prob), p=action_prob) # Choose a random action
