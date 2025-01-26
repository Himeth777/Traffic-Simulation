import pickle
import numpy as np

# If you want to reuse the same agent class or data structures:
from model import QLearningAgent, control_traffic, n_lanes  # or replicate them

# Recreate or reference your agent
agent = QLearningAgent(n_lanes=n_lanes, max_duration=40)  # example arguments

# Load Q-table from file
with open("tables/q_table.pkl", "rb") as f:
    agent.q_table = pickle.load(f)  # Now agent.q_table has the previously trained data

# Now you can call control_traffic with the loaded agent
test_scenario = np.array([30, 32, 31, 24])
lane_durations = control_traffic(test_scenario, agent)
print("Loaded Q-table. Recommended durations for test_scenario:", lane_durations)