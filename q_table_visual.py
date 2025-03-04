import streamlit as st
import pickle

def main():
    st.title("Q-Table Visualization")
    
    # Load your Q-table
    with open("tables/q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    
    # Display the Q-table
    st.write("Q-Table:", q_table)

if __name__ == "__main__":
    main()