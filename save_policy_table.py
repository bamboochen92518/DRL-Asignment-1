import json
import numpy as np
import itertools
import random


def save_policy_table(policy_table):
    new_table = dict()
    for state in policy_table.keys():
        p = policy_table[state]
        new_table[str(state)] = p.tolist()
    with open('policy.json', 'w') as f:
        json.dump(new_table, f)
