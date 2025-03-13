import json
import numpy as np
import itertools


def save_policy_table(policy_table):
    new_table = dict()
    for state in policy_table.keys():
        p = policy_table[state]
        r = range(2)
        m = np.min(p)
        for i, j, k, l in itertools.product(range(2), repeat=4):
            ns = (state[0], state[1], i, j, k, l)
            pp = [m if i else p[0], m if j else p[1], m if k else p[2], m if l else p[3]]
            new_table[str(ns)] = pp
    with open('policy.json', 'w') as f:
        json.dump(new_table, f)
