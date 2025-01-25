import numpy as np
from collections import defaultdict
import random
import pickle

def initialize_q_table(n_lanes, max_duration, min_duration):
    return np.zeros((n_lanes, max_duration - min_duration + 1))

class TrafficJunction:
    def __init__(self):
        self.lanes = 4  # 4-way intersection
        self.max_vehicles = 20  # Maximum vehicles per lane
        self.min_duration = 10  # Minimum green light duration in seconds
        self.max_duration = 40  # Maximum green light duration in seconds
        self.reset()
    
    def reset(self):
        # Random number of vehicles in each lane
        self.vehicles = np.random.randint(0, self.max_vehicles, self.lanes)
        return self._get_state()
    
    def _get_state(self):
        # Convert vehicle counts to a discrete state representation (5 levels: 0-4)
        return tuple(min(4, v // 5) for v in self.vehicles)
    
    def step(self, action):
        # `action` now contains both lane and duration
        lane, duration = action
        duration = max(self.min_duration, min(duration, self.max_duration))  # Clamp duration
        
        # Clear vehicles in the chosen lane based on duration
        vehicles_cleared = min(self.vehicles[lane], duration // 2)
        self.vehicles[lane] -= vehicles_cleared
        
        # Add new vehicles to all lanes
        self.vehicles += np.random.randint(0, 5, self.lanes)
        self.vehicles = np.clip(self.vehicles, 0, self.max_vehicles)
        
        # Calculate reward with improvements:
        reward = -np.sum(self.vehicles)  # Negative total waiting
        
        # Bonus for clearing vehicles
        reward += vehicles_cleared * 2  # Reward for clearing vehicles
        
        # Penalize congestion more when vehicles in any lane exceed 15
        penalty = 0
        for v in self.vehicles:
            if v > 15:
                penalty -= 50  # Strong penalty for high congestion
        reward += penalty
        
        # Penalize lanes that are left unused (zero vehicles after green light)
        for i in range(self.lanes):
            if self.vehicles[i] == 0:
                penalty -= 10  # Penalty for underutilizing a lane
        reward += penalty
        
        # Reward for balancing traffic flow across lanes
        lane_vehicle_cleared = np.sum(self.vehicles) - np.sum(self.vehicles)
        reward += lane_vehicle_cleared * 0.5  # Reward based on clearing vehicles evenly
        
        return self._get_state(), reward


class QLearningAgent:
    def __init__(self, n_lanes, max_duration, learning_rate=0.05, discount_factor=0.99):
        self.n_lanes = n_lanes
        self.max_duration = max_duration
        self.min_duration = 10  # Minimum green light duration
        self.q_table = dict()  # Replace defaultdict with regular dict
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = 0.9
        self.state_visits = defaultdict(int)
    
    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = initialize_q_table(self.n_lanes, self.max_duration, self.min_duration)
        return self.q_table[state]
    
    def get_action(self, state, episode, max_episodes):
        self.epsilon = 0.1 + (0.9 - 0.1) * np.exp(-episode / (max_episodes / 10))
        if random.random() < self.epsilon:
            # Explore: Randomly choose lane and duration
            lane = random.randint(0, self.n_lanes - 1)
            duration = random.randint(self.min_duration, self.max_duration)
        else:
            # Exploit: Choose the best lane and duration
            q_values = self.get_q_values(state)
            best_action = np.unravel_index(
                np.argmax(q_values), (self.n_lanes, self.max_duration - self.min_duration + 1)
            )
            lane, duration_index = best_action
            duration = duration_index + self.min_duration  # Map index back to duration
        return lane, duration
    
    def learn(self, state, action, reward, next_state):
        lane, duration = action
        duration_index = duration - self.min_duration  # Map duration to index
        current_q = self.get_q_values(state)[lane][duration_index]
        max_future_q = np.max(self.get_q_values(next_state))  # Max Q-value for the next state
        td_target = reward + self.gamma * max_future_q
        td_error = td_target - current_q
        self.q_table[state][lane][duration_index] += self.lr * td_error


# Training the model
env = TrafficJunction()
n_lanes = 4
max_duration = 40  # Maximum green light duration
agent = QLearningAgent(n_lanes=n_lanes, max_duration=max_duration)
episodes = 2000
steps_per_episode = 1000
max_episodes = episodes

def train_agent():
    """
    Runs Q-learning episodes, tracks coverage, and saves the Q-table upon completion.
    Make sure 'agent', 'env', 'visited_states', 'episodes', 'steps_per_episode',
    and 'max_episodes' are defined globally or adjust them as parameters here.
    """
    global agent, env, visited_states, episodes, steps_per_episode, max_episodes

    visited_states = set()  # Track which states have been seen

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(steps_per_episode):
            # Record the current state
            visited_states.add(state)

            # Increment visit count (if your agent tracks visits)
            agent.state_visits[state] += 1

            # Epsilon-greedy action selection
            action = agent.get_action(state, episode, max_episodes)

            # Environment transition
            next_state, reward = env.step(action)

            # Q-learning update
            agent.learn(state, action, reward, next_state)

            # Move on to next step
            state = next_state
            total_reward += reward

        # Periodic reporting
        if episode % 100 == 0:
            coverage = len(visited_states) / 625 * 100  # 625 = 5×5×5×5 if states are (0..4)^4
            print(f"Episode {episode}, Total Reward: {total_reward}")
            print(f"Visited States: {len(visited_states)}/625 ({coverage:.2f}% coverage)")

        # Early stop if all states have been covered
        if len(visited_states) == 625:
            print(f"Achieved 100% state coverage by episode {episode}")
            break

    # Save Q-table to file
    with open("tables/q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    print("Q-table saved to tables/q_table.pkl")

def control_traffic(vehicle_counts, agent):
    """
    Returns a recommended duration for each of the 4 lanes.
    """
    # Convert vehicle counts to discrete state representation
    state = tuple(min(4, v // 5) for v in vehicle_counts)
    
    # Get Q-values for this state
    q_values = agent.q_table.get(state)
    if q_values is None:
        q_values = initialize_q_table(agent.n_lanes, agent.max_duration, agent.min_duration)
        agent.q_table[state] = q_values
    
    recommended_durations = []
    for lane_idx in range(agent.n_lanes):
        # Get the best duration index for this lane
        best_duration_idx = np.argmax(q_values[lane_idx])
        # Convert index to actual duration
        recommended_duration = best_duration_idx + agent.min_duration
        recommended_durations.append(recommended_duration)
    
    return list(map(int, recommended_durations))

if __name__ == "__main__":
    train_agent()


