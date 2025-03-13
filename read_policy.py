import pickle
import json

# Load from a pickle file (binary mode)
with open('policy.pkl', 'rb') as f:
    policy_table = pickle.load(f)

for i in policy_table.keys():
    print(i, policy_table[i])
