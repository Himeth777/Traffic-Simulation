import pickle
import numpy as np
import json
import os
from model import QLearningAgent, control_traffic, n_lanes

def load_detection_results(results_file="detection_results.json"):
    """Load vehicle counts from detection results"""
    try:
        with open(os.path.join("results", results_file), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No detection results found at {results_file}")
        return None

def get_lane_timings(video_counts):
    """Get recommended timings based on vehicle counts from each video/lane"""
    agent = QLearningAgent(n_lanes=n_lanes, max_duration=40)
    with open("tables/q_table.pkl", "rb") as f:
        agent.q_table = pickle.load(f)

    # Map videos to their respective lanes
    lane_counts = np.zeros(4)
    video_to_lane = {
        "video1.mp4": 0,  # Horizontal incoming
        "video2.mp4": 1,  # Horizontal outgoing
        "video3.mp4": 2,  # Vertical incoming
        "video4.mp4": 3   # Vertical outgoing
    }
    
    # Fill lane counts from video counts
    for video_path, count in video_counts.items():
        video_name = os.path.basename(video_path)
        if video_name in video_to_lane:
            lane_counts[video_to_lane[video_name]] = count

    # Get recommended durations using actual lane counts
    durations = control_traffic(lane_counts, agent)
    return durations

def main():
    results = load_detection_results()
    if not results:
        print("No detection results available")
        return

    print("\nLane Vehicle Counts and Recommended Timings:")
    print("------------------------------------------")
    
    # Get counts for each video/lane
    video_counts = {path: data['total_count'] for path, data in results.items()}
    
    # Print detected vehicles per lane
    for video_path, count in video_counts.items():
        print(f"Lane {os.path.basename(video_path)}: {count} vehicles")
    
    # Get timings based on actual lane counts
    timings = get_lane_timings(video_counts)
    
    print("\nRecommended green light durations:")
    print(f"Horizontal incoming (video1): {timings[0]} seconds")
    print(f"Horizontal outgoing (video2): {timings[1]} seconds")
    print(f"Vertical incoming   (video3): {timings[2]} seconds")
    print(f"Vertical outgoing   (video4): {timings[3]} seconds")

if __name__ == "__main__":
    main()