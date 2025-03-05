import streamlit as st
import pickle
import pandas as pd
import numpy as np

def main():
    st.title("Q-Table Visualization")
    
    try:
        # Load your Q-table
        with open("tables/q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
        
        # Check what type of data structure the Q-table is
        if isinstance(q_table, dict):
            visualize_dict_qtable(q_table)
        elif isinstance(q_table, np.ndarray):
            visualize_array_qtable(q_table)
        else:
            st.error(f"Unexpected Q-table type: {type(q_table)}")
            st.write("Expected either a dictionary or numpy array.")
    
    except Exception as e:
        st.error(f"Error loading or displaying Q-table: {e}")
        st.write("Please ensure the Q-table file exists and is properly formatted.")

def visualize_dict_qtable(q_table):
    """Visualize dictionary-based Q-table"""
    # Convert the Q-table tuple keys to strings to make it JSON serializable
    serializable_q_table = {}
    for key, value in q_table.items():
        # Convert tuple key to string representation
        str_key = str(key)
        serializable_q_table[str_key] = value
    
    # Display high-level statistics
    st.subheader("Q-Table Statistics (Dictionary Format)")
    st.write(f"Number of states: {len(serializable_q_table)}")
    
    # Extract all unique actions
    all_actions = []
    for values in q_table.values():
        all_actions.extend(values.keys())
    unique_actions = sorted(list(set(all_actions)))
    st.write(f"Unique actions: {unique_actions}")
    
    # Display Q-values in a structured format
    st.subheader("Q-Table Values")
    
    # Convert to a format better for display
    data = []
    for state, actions in q_table.items():
        row = {"State": str(state)}
        # Add Q-values for each action
        for action in unique_actions:
            row[f"Action {action}"] = actions.get(action, 0)
        data.append(row)
    
    # Create a DataFrame for better display
    df = pd.DataFrame(data)
    
    # Display the DataFrame
    st.dataframe(df)
    
    # Show a bar chart of average Q-values
    display_value_chart(df)

def visualize_array_qtable(q_table):
    """Visualize array-based Q-table"""
    st.subheader("Q-Table Statistics (Array Format)")
    
    # Get dimensions
    shape = q_table.shape
    st.write(f"Q-table shape: {shape}")
    
    if len(shape) == 2:
        # Typical case: states x actions
        states = shape[0]
        actions = shape[1]
        st.write(f"Number of states: {states}")
        st.write(f"Number of actions: {actions}")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(q_table)
        
        # Rename columns to show action numbers
        df.columns = [f"Action {i}" for i in range(actions)]
        df.index.name = "State"
        
        # Display the Q-table
        st.subheader("Q-Table Values")
        st.dataframe(df)
        
        # Display statistics
        st.subheader("Q-Value Statistics")
        st.write(f"Min Q-value: {np.min(q_table)}")
        st.write(f"Max Q-value: {np.max(q_table)}")
        st.write(f"Mean Q-value: {np.mean(q_table)}")
        
        # Display chart
        display_value_chart(df)
    else:
        st.write("This appears to be a multi-dimensional Q-table.")
        st.write("Showing summary statistics instead of full table.")
        st.write(f"Min Q-value: {np.min(q_table)}")
        st.write(f"Max Q-value: {np.max(q_table)}")
        st.write(f"Mean Q-value: {np.mean(q_table)}")
        
        # Show a sample of the data
        st.subheader("Sample Q-values")
        flattened = q_table.reshape(-1, q_table.shape[-1])
        sample_size = min(100, flattened.shape[0])
        sample_indices = np.random.choice(flattened.shape[0], sample_size, replace=False)
        sample = flattened[sample_indices]
        sample_df = pd.DataFrame(sample, columns=[f"Action {i}" for i in range(q_table.shape[-1])])
        st.dataframe(sample_df)

def display_value_chart(df):
    """Display charts for Q-values"""
    # Get only action columns
    action_columns = [col for col in df.columns if col.startswith("Action")]
    
    # Show a bar chart of average Q-values
    st.subheader("Average Q-Value by Action")
    st.bar_chart(df[action_columns].mean())
    
    # Show a sample of values as a heatmap
    st.subheader("Q-Value Sample Distribution")
    sample_size = min(50, len(df))  # Limit to a reasonable sample size
    sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
    st.write(f"Showing sample of {len(sample_df)} states")
    
    # Get descriptive statistics
    stats_df = df[action_columns].describe()
    st.dataframe(stats_df)

if __name__ == "__main__":
    main()