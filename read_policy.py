import json

# Load from a pickle file (binary mode)
with open('policy.json', 'r') as f:
    policy_table = json.load(f)

s = sorted(policy_table.keys())

for ss in s:
    print(ss)
print(len(policy_table.keys()))
