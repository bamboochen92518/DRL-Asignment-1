# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym


is_picked = False
visited = {"R": 0, "G": 0, "Y": 0, "B": 0}
destination = None


def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    # Load global variable
    global is_picked
    global visited
    global destination

    action_prob = [1, 1, 1, 1, 1, 1]

    taxi_row, taxi_col, Rx, Ry, Gx, Gy, Yx, Yy, Bx, By, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    destination_to_loc = {'R': (Rx, Ry), 'G': (Gx, Gy), 'Y': (Yx, Yy), 'B': (Bx, By)}

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

    # If able to finish
    if is_picked and destination_look and on_location():
        return 5

    # Remove invalid option first
    if obstacle_south:
        action_prob[0] = 0
    if obstacle_north:
        action_prob[1] = 0
    if obstacle_east:
        action_prob[2] = 0
    if obstacle_west:
        action_prob[3] = 0
    if not passenger_look or is_picked or not on_location():
        action_prob[4] = 0
    if not destination_look or not on_location():
        action_prob[5] = 0

    # If on_location, modify visited dict
    if on_location() is not None:
        visited[on_location()] += 1

        if passenger_look and not is_picked:
            is_picked = True
            visited['R'] = 1
            visited['G'] = 1
            visited['Y'] = 1
            visited['B'] = 1
            return 4

        if destination_look:
            destination = on_location()

    # Determine where to go
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

    direction = (taxi_row - target[0], taxi_col - target[1])
    
    scale = 2
    if direction[0] > 0:
        action_prob[1] *= scale
    if direction[0] < 0:
        action_prob[0] *= scale
    if direction[1] > 0:
        action_prob[3] *= scale
    if direction[1] < 0:
        action_prob[2] *= scale

    action_prob = np.array(action_prob, dtype=np.float32)
    action_prob = action_prob / action_prob.sum()
    
    return np.random.choice(len(action_prob), p=action_prob) # Choose a random action
