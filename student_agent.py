# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    possible_action = [0, 1, 2, 3, 4, 5]

    taxi_row, taxi_col, Rx, Ry, Gx, Gy, Yx, Yy, Bx, By, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if obstacle_south:
        possible_action.remove(0)
    if obstacle_north:
        possible_action.remove(1)
    if obstacle_east:
        possible_action.remove(2)
    if obstacle_west:
        possible_action.remove(3)
    if not passenger_look:
        possible_action.remove(4)
    if not destination_look:
        possible_action.remove(5)

    return random.choice(possible_action) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

