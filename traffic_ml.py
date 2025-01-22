import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

class TrafficLightOptimizer:
    def __init__(self):
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = {}
        self.actions = [5000, 10000, 15000]  # Light timings in ms
        
    def get_state(self, counters):
        # Convert counter values to state representation
        return (
            counters['horizontal']['incoming'],
            counters['horizontal']['outgoing'],
            counters['vertical']['incoming'],
            counters['vertical']['outgoing']
        )
    
    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
            
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def update(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}
            
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values())
        
        # Q-learning update
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )

# Initialize optimizer
optimizer = TrafficLightOptimizer()

@app.route('/optimize', methods=['POST'])
def optimize_timing():
    data = request.json
    counters = data['counters']
    
    # Get current state
    state = optimizer.get_state(counters)
    
    # Calculate reward (negative of total queue length)
    total_queued = sum(sum(lane.values()) for lane in counters.values())
    reward = -total_queued
    
    # Get optimal timing
    optimal_timing = optimizer.get_action(state)
    
    # Save Q-table periodically
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(optimizer.q_table, f)
    
    return jsonify({'timing': optimal_timing})

if __name__ == '__main__':
    # Load existing Q-table if available
    try:
        with open('q_table.pkl', 'rb') as f:
            optimizer.q_table = pickle.load(f)
    except FileNotFoundError:
        pass
    
    app.run(port=5000)