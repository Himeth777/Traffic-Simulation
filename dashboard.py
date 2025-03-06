import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageDraw, ImageFont
import os
from output import load_detection_results, get_lane_timings

class SimVehicle:
    def __init__(self, x, y, direction, lane, color):
        self.x = x
        self.y = y
        self.direction = direction  # "horizontal" or "vertical"
        self.lane = lane  # "incoming" or "outgoing"
        self.color = color
        self.speed = 5
        self.size = 20  # Size of vehicle
        self.stopped = False
        self.passed_light = False  # Track if vehicle has passed the traffic light
        self.min_vehicle_gap = 45  # Minimum gap between vehicles when stopped
        self.exited = False  # Track if vehicle has exited the screen

    def move(self, active_light, width, height, all_vehicles):
        # Store previous position to detect when vehicle exits screen
        prev_x, prev_y = self.x, self.y
        
        # If vehicle has already passed its traffic light, don't check for stopping
        if not self.passed_light:
            # Check if this vehicle should stop at a traffic light
            should_stop = self.check_traffic_light(active_light, width, height)
            
            # Check if there's a vehicle ahead to avoid collision
            vehicle_ahead = self.check_vehicle_ahead(all_vehicles)
            
            if should_stop or vehicle_ahead:
                self.stopped = True
                return
        
        self.stopped = False
        
        # Move based on direction and lane
        if self.direction == "horizontal":
            if self.lane == "incoming":  # Moving right
                self.x += self.speed
                if self.x > width + 50:  # Reset if off screen
                    self.x = -50
                    self.passed_light = False
            else:  # Moving left
                self.x -= self.speed
                if self.x < -50:  # Reset if off screen
                    self.x = width + 50
                    self.passed_light = False
        else:  # Vertical
            if self.lane == "incoming":  # Moving down
                self.y += self.speed
                if self.y > height + 50:  # Reset if off screen
                    self.y = -50
                    self.passed_light = False
            else:  # Moving up
                self.y -= self.speed
                if self.y < -50:  # Reset if off screen
                    self.y = height + 50
                    self.passed_light = False

        # Detect when vehicle exits the screen
        exited = False
        if self.direction == "horizontal":
            if self.lane == "incoming" and prev_x < width and self.x >= width:
                exited = True  # Vehicle exited right side
            elif self.lane == "outgoing" and prev_x > 0 and self.x <= 0:
                exited = True  # Vehicle exited left side
        else:  # vertical
            if self.lane == "incoming" and prev_y < height and self.y >= height:
                exited = True  # Vehicle exited bottom
            elif self.lane == "outgoing" and prev_y > 0 and self.y <= 0:
                exited = True  # Vehicle exited top
        
        return exited

    def check_traffic_light(self, active_light, width, height):
        """Check if vehicle should stop at a traffic light"""
        # Map traffic lights to directions, lanes and positions
        center_x = width / 2
        center_y = height / 2
        road_width = 100
        
        # Define traffic light positions
        traffic_map = {
            1: {
                "direction": "horizontal", 
                "lane": "incoming",
                "position": center_x - road_width / 2
            },
            2: {
                "direction": "horizontal", 
                "lane": "outgoing",
                "position": center_x + road_width / 2
            },
            3: {
                "direction": "vertical", 
                "lane": "incoming",
                "position": center_y - road_width / 2
            },
            4: {
                "direction": "vertical", 
                "lane": "outgoing",
                "position": center_y + road_width / 2
            }
        }
        
        # Check if vehicle is approaching its traffic light
        for light_id, light_info in traffic_map.items():
            if light_info["direction"] == self.direction and light_info["lane"] == self.lane:
                # If this is the active light, vehicle can proceed
                if light_id == active_light:
                    return False
                
                # Check if vehicle has passed the light already
                if self.direction == "horizontal":
                    if self.lane == "incoming":
                        # Vehicle going right
                        if self.x >= light_info["position"]:
                            self.passed_light = True
                            return False
                        else:
                            # Stop only if approaching the light
                            return self.x > light_info["position"] - 100 and self.x < light_info["position"]
                    else:
                        # Vehicle going left
                        if self.x <= light_info["position"]:
                            self.passed_light = True
                            return False
                        else:
                            # Stop only if approaching the light
                            return self.x < light_info["position"] + 100 and self.x > light_info["position"]
                else:  # Vertical
                    if self.lane == "incoming":
                        # Vehicle going down
                        if self.y >= light_info["position"]:
                            self.passed_light = True
                            return False
                        else:
                            # Stop only if approaching the light
                            return self.y > light_info["position"] - 100 and self.y < light_info["position"]
                    else:
                        # Vehicle going up
                        if self.y <= light_info["position"]:
                            self.passed_light = True
                            return False
                        else:
                            # Stop only if approaching the light
                            return self.y < light_info["position"] + 100 and self.y > light_info["position"]
        
        return False  # Default to not stopping if no matching light

    def check_vehicle_ahead(self, all_vehicles):
        """Check if there's a vehicle ahead that this vehicle should stop behind"""
        for other in all_vehicles:
            # Skip self or vehicles in different lanes
            if other is self or other.direction != self.direction or other.lane != self.lane:
                continue
                
            # Check based on direction and lane
            if self.direction == "horizontal":
                if self.lane == "incoming":
                    # Vehicle going right
                    if other.x > self.x and other.x - self.x < self.min_vehicle_gap:
                        return True
                else:
                    # Vehicle going left
                    if other.x < self.x and self.x - other.x < self.min_vehicle_gap:
                        return True
            else:  # vertical
                if self.lane == "incoming":
                    # Vehicle going down
                    if other.y > self.y and other.y - self.y < self.min_vehicle_gap:
                        return True
                else:
                    # Vehicle going up
                    if other.y < self.y and self.y - other.y < self.min_vehicle_gap:
                        return True
                        
        return False

def draw_traffic_junction(active_light=None, width=600, height=600, vehicles=None):
    """Draw a traffic junction with optional active traffic light and vehicles"""
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Roads (gray)
    road_width = 100
    draw.rectangle([width/2 - road_width/2, 0, width/2 + road_width/2, height], fill='gray')  # Vertical road
    draw.rectangle([0, height/2 - road_width/2, width, height/2 + road_width/2], fill='gray')  # Horizontal road
    
    # Intersection (dark gray)
    draw.rectangle([width/2 - road_width/2, height/2 - road_width/2, 
                   width/2 + road_width/2, height/2 + road_width/2], fill='darkgray')
    
    # Lane dividers (white lines)
    # Vertical road
    draw.line([width/2, 0, width/2, height], fill='white', width=3)
    # Horizontal road
    draw.line([0, height/2, width, height/2], fill='white', width=3)
    
    # Traffic lights
    lights = [
        {"name": "Horizontal incoming", "id": 1, "pos": (width/2 - road_width - 30, height/2 - road_width/4 - 30)},
        {"name": "Horizontal outgoing", "id": 2, "pos": (width/2 + road_width + 30, height/2 + road_width/4)},
        {"name": "Vertical incoming", "id": 3, "pos": (width/2 + road_width/4, height/2 - road_width - 30)},
        {"name": "Vertical outgoing", "id": 4, "pos": (width/2 - road_width/4 - 30, height/2 + road_width + 30)}
    ]
    
    # Draw traffic lights
    for light in lights:
        # Traffic light box
        x, y = light["pos"]
        draw.rectangle([x, y, x + 30, y + 60], fill='black', outline='white')
        
        # Red light
        if active_light != light["id"]:
            draw.ellipse([x + 5, y + 5, x + 25, y + 25], fill='red')
            draw.ellipse([x + 5, y + 35, x + 25, y + 55], fill='darkgray')
        else:
            # Green light
            draw.ellipse([x + 5, y + 5, x + 25, y + 25], fill='darkgray')
            draw.ellipse([x + 5, y + 35, x + 25, y + 55], fill='green')
        
        # Label
        draw.text((x, y - 20), light["name"], fill='black')
    
    # Draw vehicles if provided
    if vehicles:
        for v in vehicles:
            # Draw the vehicle as a rectangle with its color
            if v.direction == "horizontal":
                # Horizontal vehicles are longer rectangles
                draw.rectangle([v.x, v.y, v.x + v.size*1.5, v.y + v.size], fill=v.color, outline='black')
            else:
                # Vertical vehicles are taller rectangles
                draw.rectangle([v.x, v.y, v.x + v.size, v.y + v.size*1.5], fill=v.color, outline='black')
    
    return image

def initialize_vehicles(vehicle_counts, width, height):
    """Initialize vehicles based on the real counts from detection"""
    vehicles = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    road_width = 100
    center_x = width / 2
    center_y = height / 2
    
    # Define lane positions - adjusted to be behind traffic lights
    lanes = {
        "horizontal": {
            "incoming": center_y - road_width/4 - 10,  # Top half of bottom horizontal lane
            "outgoing": center_y + road_width/4 + 10   # Bottom half of top horizontal lane
        },
        "vertical": {
            "incoming": center_x - road_width/4 - 10,   # Left half of right vertical lane
            "outgoing": center_x + road_width/4 + 10    # Right half of left vertical lane
        }
    }
    
    # Define start positions to be behind traffic lights
    start_positions = {
        "horizontal": {
            "incoming": [0, center_x - road_width - 50],
            "outgoing": [width, center_x + road_width + 50]
        },
        "vertical": {
            "incoming": [0, center_y - road_width - 50],
            "outgoing": [height, center_y + road_width + 50]
        }
    }
    
    # Create vehicles for each lane based on counts
    for i, (video, count) in enumerate(vehicle_counts.items()):
        direction = "horizontal" if i < 2 else "vertical"
        lane = "incoming" if i % 2 == 0 else "outgoing"
        
        # Scale display count while keeping original count for tracking
        display_count = min(int(count / 10) + 1, 8)  # Scale down and ensure at least 1
        
        for j in range(display_count):
            # Distribute vehicles evenly along their respective roads, but behind traffic lights
            if direction == "horizontal":
                y = lanes[direction][lane]
                if lane == "incoming":
                    # Start from left side, behind traffic light
                    x = start_positions[direction][lane][0] + j * 80
                    if x >= start_positions[direction][lane][1]:
                        x = start_positions[direction][lane][0] + (j % 3) * 80
                else:
                    # Start from right side, behind traffic light
                    x = start_positions[direction][lane][0] - j * 80
                    if x <= start_positions[direction][lane][1]:
                        x = start_positions[direction][lane][0] - (j % 3) * 80
            else:  # vertical
                x = lanes[direction][lane]
                if lane == "incoming":
                    # Start from top, behind traffic light
                    y = start_positions[direction][lane][0] + j * 80
                    if y >= start_positions[direction][lane][1]:
                        y = start_positions[direction][lane][0] + (j % 3) * 80
                else:
                    # Start from bottom, behind traffic light
                    y = start_positions[direction][lane][0] - j * 80
                    if y <= start_positions[direction][lane][1]:
                        y = start_positions[direction][lane][0] - (j % 3) * 80
            
            color = colors[j % len(colors)]
            vehicles.append(SimVehicle(x, y, direction, lane, color))
    
    return vehicles

def count_vehicles_by_lane(vehicles):
    """Count vehicles in each lane"""
    counts = {
        "Horizontal incoming": 0,
        "Horizontal outgoing": 0,
        "Vertical incoming": 0,
        "Vertical outgoing": 0
    }
    
    for vehicle in vehicles:
        lane_key = f"{vehicle.direction.capitalize()} {vehicle.lane}"
        counts[lane_key] += 1
        
    return counts

def spawn_new_vehicle(direction, lane, width, height, colors):
    """Create a new vehicle at the edge of the frame"""
    road_width = 100
    center_x = width / 2
    center_y = height / 2
    
    # Define lane Y positions
    lanes = {
        "horizontal": {
            "incoming": center_y - road_width/4 - 10,  # Top half of bottom horizontal lane
            "outgoing": center_y + road_width/4 + 10   # Bottom half of top horizontal lane
        },
        "vertical": {
            "incoming": center_x - road_width/4 - 10,   # Left half of right vertical lane
            "outgoing": center_x + road_width/4 + 10    # Right half of left vertical lane
        }
    }
    
    # Define starting positions at edges of frame
    if direction == "horizontal":
        y = lanes[direction][lane]
        if lane == "incoming":
            x = -50  # Start from left edge
        else:
            x = width + 50  # Start from right edge
    else:  # vertical
        x = lanes[direction][lane]
        if lane == "incoming":
            y = -50  # Start from top edge
        else:
            y = height + 50  # Start from bottom edge
    
    # Choose random color
    color = colors[np.random.randint(0, len(colors))]
    
    return SimVehicle(x, y, direction, lane, color)

def main():
    st.set_page_config(page_title="Traffic Junction Dashboard", layout="wide")
    st.title("Traffic Junction Dashboard")
    
    # Load detection results from output.py
    results = load_detection_results()
    if not results:
        st.error("No detection results available. Please run detection first.")
        return
        
    # Get vehicle counts for each lane
    video_counts = {os.path.basename(path): data['total_count'] for path, data in results.items()}
    
    # Create a mapping for better display
    lane_mapping = {
        "video1.mp4": "Horizontal incoming",
        "video2.mp4": "Horizontal outgoing",
        "video3.mp4": "Vertical incoming", 
        "video4.mp4": "Vertical outgoing"
    }
    
    # Create a DataFrame for display
    data = []
    for video, count in video_counts.items():
        lane = lane_mapping.get(video, video)
        data.append({"Lane": lane, "Video": video, "Vehicle Count": count})
    
    df = pd.DataFrame(data)
    
    # Calculate timings based on vehicle counts
    timings = get_lane_timings({path: video_counts[os.path.basename(path)] for path in results.keys()})
    
    # Create timing data for display
    timing_data = []
    for i, (lane, video) in enumerate(zip(
        ["Horizontal incoming", "Horizontal outgoing", "Vertical incoming", "Vertical outgoing"],
        ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
    )):
        timing_data.append({
            "Lane": lane,
            "Video": video,
            "Duration (seconds)": timings[i]
        })
    
    timing_df = pd.DataFrame(timing_data)
    
    # Create 2 columns for the dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vehicle Counts by Lane")
        # Bar chart of vehicle counts
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df["Lane"], df["Vehicle Count"], color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        ax.set_ylabel("Vehicle Count")
        ax.set_title("Number of Vehicles Detected in Each Lane")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        
        # Show the raw data
        st.dataframe(df)
    
    with col2:
        st.subheader("Recommended Green Light Durations")
        # Bar chart of green light durations
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(timing_df["Lane"], timing_df["Duration (seconds)"], color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        ax.set_ylabel("Duration (seconds)")
        ax.set_title("Recommended Green Light Duration for Each Lane")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        
        # Show the raw data
        st.dataframe(timing_df)
    
    # Traffic light simulation
    st.header("Traffic Light Simulation")
    
    # Start button for running both simulations simultaneously
    if st.button("Run Both Simulations Side-by-Side", key="both_simulations"):
        run_simultaneous_simulations(results, video_counts, timings)
    
    # Or run individual simulations 
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        st.subheader("AI-Optimized Simulation")
        if st.button("Start Optimized Simulation", key="optimized_sim"):
            run_simulation(results, video_counts, timings, "Optimized")
    
    with sim_col2:
        st.subheader("Default Timing Simulation")
        if st.button("Start Default Simulation (10s each)", key="default_sim"):
            # Use fixed timings of 10 seconds for each light
            default_timings = [10, 10, 10, 10]
            run_simulation(results, video_counts, default_timings, "Default 10s")
    
    # Create a section for additional insights
    st.header("Traffic Analysis Insights")
    
    # Calculate and show the lane with the most traffic
    busiest_lane = df.loc[df["Vehicle Count"].idxmax()]
    st.info(f"Busiest lane: {busiest_lane['Lane']} with {busiest_lane['Vehicle Count']} vehicles")
    
    # Calculate and show the lane with the least traffic
    quietest_lane = df.loc[df["Vehicle Count"].idxmin()]
    st.info(f"Quietest lane: {quietest_lane['Lane']} with {quietest_lane['Vehicle Count']} vehicles")
    
    # Calculate traffic imbalance
    max_count = df["Vehicle Count"].max()
    min_count = df["Vehicle Count"].min()
    imbalance = (max_count - min_count) / max_count if max_count > 0 else 0
    st.info(f"Traffic imbalance: {imbalance:.1%}")
    
    # Relationship between vehicle count and green light duration
    st.subheader("Relationship Between Vehicle Count and Green Light Duration")
    
    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["Vehicle Count"], timing_df["Duration (seconds)"], s=100, alpha=0.7)
    
    # Add labels for each point
    for i, lane in enumerate(df["Lane"]):
        ax.annotate(lane, 
                   (df["Vehicle Count"].iloc[i], timing_df["Duration (seconds)"].iloc[i]),
                   textcoords="offset points",
                   xytext=(0,10), 
                   ha='center')
    
    ax.set_xlabel("Vehicle Count")
    ax.set_ylabel("Green Light Duration (seconds)")
    ax.set_title("Vehicle Count vs Green Light Duration")
    ax.grid(True, alpha=0.3)
    
    # Add a best fit line
    x = df["Vehicle Count"]
    y = timing_df["Duration (seconds)"]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.7, label=f"Trend: y={z[0]:.2f}x + {z[1]:.2f}")
    ax.legend()
    
    st.pyplot(fig)

def run_simulation(results, video_counts, timings, mode_name):
    """Run a traffic light simulation with the given timings"""
    # Create placeholders for simulation displays
    simulation_placeholder = st.empty()
    vehicle_count_placeholder = st.empty()
    cycle_info_placeholder = st.empty()
    cycle_summary_placeholder = st.empty()
    
    # Lane traffic rates (vehicles per second) based on detection data
    lane_tracking = {
        "Horizontal incoming": {"initial": video_counts.get("video1.mp4", 0), "cleared": 0},
        "Horizontal outgoing": {"initial": video_counts.get("video2.mp4", 0), "cleared": 0},
        "Vertical incoming": {"initial": video_counts.get("video3.mp4", 0), "cleared": 0},
        "Vertical outgoing": {"initial": video_counts.get("video4.mp4", 0), "cleared": 0}
    }
    
    lane_traffic_rates = {
        "Horizontal incoming": video_counts.get("video1.mp4", 0) / 300,  # Assume 5-minute video
        "Horizontal outgoing": video_counts.get("video2.mp4", 0) / 300,
        "Vertical incoming": video_counts.get("video3.mp4", 0) / 300,
        "Vertical outgoing": video_counts.get("video4.mp4", 0) / 300
    }
    
    total_initial_vehicles = sum(data["initial"] for data in lane_tracking.values())
    
    # Initialize simulation state
    cycle_count = 0
    keep_running = True
    cycle_history = []
    
    # Create initial vehicles
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    vehicles = []
    
    # Add initial vehicles (about 1/4 of the total count per lane)
    for direction in ["horizontal", "vertical"]:
        for lane in ["incoming", "outgoing"]:
            lane_key = f"{direction.capitalize()} {lane}"
            initial_count = int(lane_tracking[lane_key]["initial"] / 4)
            for _ in range(initial_count):
                vehicles.append(spawn_new_vehicle(direction, lane, 600, 600, colors))
    
    # Variables to track vehicle spawning
    last_spawn_time = time.time()
    
    # Track cleared vehicles across all cycles
    total_vehicles_cleared = 0
    
    # Display mode name
    st.subheader(f"Running {mode_name} Simulation")
    
    # Run simulation for multiple cycles
    while keep_running:
        cycle_count += 1
        cycle_info_placeholder.info(f"Running Cycle {cycle_count}")
        
        # Track vehicles cleared in this cycle
        vehicles_cleared_this_cycle = {
            "Horizontal incoming": 0,
            "Horizontal outgoing": 0,
            "Vertical incoming": 0,
            "Vertical outgoing": 0
        }
        
        # Set up cycle timer
        cycle_time = sum(timings)
        cycle_progress = st.progress(0)
        cycle_start_time = time.time()
        
        # Run through each traffic light
        for i in range(len(timings)):
            light_id = i + 1
            duration = timings[i]
            lane_name = ["Horizontal incoming", "Horizontal outgoing", "Vertical incoming", "Vertical outgoing"][i]
            
            # Timer for this light
            light_start = time.time()
            
            # Run for the duration of this light
            while time.time() - light_start < duration:
                current_time = time.time()
                
                # Spawn new vehicles periodically based on traffic rates
                if current_time - last_spawn_time > 2.0:  # Check every 2 seconds
                    last_spawn_time = current_time
                    
                    # Try to spawn vehicles for each lane based on rates
                    for d_idx, direction in enumerate(["horizontal", "vertical"]):
                        for l_idx, lane in enumerate(["incoming", "outgoing"]):
                            lane_key = f"{direction.capitalize()} {lane}"
                            # Probability based on traffic rate
                            if np.random.random() < lane_traffic_rates[lane_key] * 2:
                                vehicles.append(spawn_new_vehicle(direction, lane, 600, 600, colors))
                
                # Process vehicle movements and exits
                vehicles_to_remove = []
                for idx, vehicle in enumerate(vehicles):
                    # Move vehicle and check if it exited
                    exited = vehicle.move(light_id, 600, 600, vehicles)
                    if exited:
                        lane_key = f"{vehicle.direction.capitalize()} {vehicle.lane}"
                        vehicles_cleared_this_cycle[lane_key] += 1
                        total_vehicles_cleared += 1
                        vehicles_to_remove.append(idx)
                
                # Remove vehicles that have exited
                for idx in sorted(vehicles_to_remove, reverse=True):
                    vehicles.pop(idx)
                
                # Count current vehicles by lane
                current_counts = count_vehicles_by_lane(vehicles)
                
                # Update display data
                table_data = []
                for lane_key in lane_tracking.keys():
                    initial = lane_tracking[lane_key]["initial"]
                    cleared_this_cycle = vehicles_cleared_this_cycle[lane_key]
                    current = current_counts.get(lane_key, 0)
                    
                    table_data.append({
                        "Lane": lane_key,
                        "Target Count": initial,
                        "Current Vehicles": current,
                        "Cleared This Cycle": cleared_this_cycle,
                        "Light Duration": timings[list(lane_tracking.keys()).index(lane_key)]
                    })
                
                # Add totals row
                total_target = sum(data["initial"] for data in lane_tracking.values())
                total_cleared_cycle = sum(vehicles_cleared_this_cycle.values())
                
                table_data.append({
                    "Lane": "TOTAL",
                    "Target Count": total_target,
                    "Current Vehicles": sum(current_counts.values()),
                    "Cleared This Cycle": total_cleared_cycle,
                    "Light Duration": "-" 
                })
                
                # Update displays
                count_df = pd.DataFrame(table_data)
                vehicle_count_placeholder.dataframe(count_df, use_container_width=True, hide_index=True)
                
                junction_image = draw_traffic_junction(active_light=light_id, vehicles=vehicles)
                simulation_placeholder.image(junction_image, use_container_width=True)
                
                # Update progress bar
                elapsed_time = time.time() - cycle_start_time
                progress = min(1.0, elapsed_time / cycle_time)
                cycle_progress.progress(progress)
                
                # Control simulation speed
                time.sleep(0.1)
        
        # After cycle completes, update cycle history
        cycle_data = {
            "Cycle": cycle_count,
            "Mode": mode_name,
            "Vehicles Cleared": sum(vehicles_cleared_this_cycle.values()),
            "Total Vehicles Cleared": total_vehicles_cleared,
            "Vehicles in Junction": len(vehicles),
            "Efficiency (vehicles/sec)": total_vehicles_cleared / (cycle_count * cycle_time)
        }
        cycle_history.append(cycle_data)
        
        # Show cycle history
        history_df = pd.DataFrame(cycle_history)
        cycle_summary_placeholder.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # Check if we should stop simulation
        if cycle_count >= 5:  # Limit to 5 cycles
            keep_running = False
            st.warning("Maximum cycle limit reached. Simulation stopped.")
    
    # Final summary
    efficiency = total_vehicles_cleared / (cycle_count * cycle_time) if cycle_count > 0 else 0
    st.success(f"""
    {mode_name} Simulation completed after {cycle_count} cycles!
    - Total vehicles cleared: {total_vehicles_cleared}
    - Remaining vehicles in junction: {len(vehicles)}
    - Efficiency: {efficiency:.4f} vehicles/second
    """)
    
    # Return data for comparison
    return {
        "mode": mode_name,
        "cycles": cycle_count,
        "cleared": total_vehicles_cleared,
        "efficiency": efficiency
    }

def run_simultaneous_simulations(results, video_counts, optimized_timings):
    """Run both optimized and default timings simulations side by side"""
    # Define default timings (10s each)
    default_timings = [10, 10, 10, 10]
    
    # Create two columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AI-Optimized Timing")
        # Create placeholders for optimized simulation
        opt_sim_placeholder = st.empty()
        opt_count_placeholder = st.empty()
        opt_cycle_info = st.empty()
        opt_cycle_summary = st.empty()
    
    with col2:
        st.subheader("Default Timing (10s each)")
        # Create placeholders for default simulation
        def_sim_placeholder = st.empty() 
        def_count_placeholder = st.empty()
        def_cycle_info = st.empty()
        def_cycle_summary = st.empty()
    
    # Initialize both simulations
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    
    # Setup tracking for optimized simulation
    opt_lane_tracking = {
        "Horizontal incoming": {"initial": video_counts.get("video1.mp4", 0), "cleared": 0},
        "Horizontal outgoing": {"initial": video_counts.get("video2.mp4", 0), "cleared": 0},
        "Vertical incoming": {"initial": video_counts.get("video3.mp4", 0), "cleared": 0},
        "Vertical outgoing": {"initial": video_counts.get("video4.mp4", 0), "cleared": 0}
    }
    
    # Setup tracking for default simulation
    def_lane_tracking = {
        "Horizontal incoming": {"initial": video_counts.get("video1.mp4", 0), "cleared": 0},
        "Horizontal outgoing": {"initial": video_counts.get("video2.mp4", 0), "cleared": 0},
        "Vertical incoming": {"initial": video_counts.get("video3.mp4", 0), "cleared": 0},
        "Vertical outgoing": {"initial": video_counts.get("video4.mp4", 0), "cleared": 0}
    }
    
    # Calculate traffic rates
    lane_traffic_rates = {
        "Horizontal incoming": video_counts.get("video1.mp4", 0) / 300,
        "Horizontal outgoing": video_counts.get("video2.mp4", 0) / 300,
        "Vertical incoming": video_counts.get("video3.mp4", 0) / 300,
        "Vertical outgoing": video_counts.get("video4.mp4", 0) / 300
    }
    
    # Create initial vehicles for optimized simulation
    opt_vehicles = []
    for direction in ["horizontal", "vertical"]:
        for lane in ["incoming", "outgoing"]:
            lane_key = f"{direction.capitalize()} {lane}"
            initial_count = int(opt_lane_tracking[lane_key]["initial"] / 4)
            for _ in range(initial_count):
                opt_vehicles.append(spawn_new_vehicle(direction, lane, 600, 600, colors))
    
    # Create initial vehicles for default simulation
    def_vehicles = []
    for direction in ["horizontal", "vertical"]:
        for lane in ["incoming", "outgoing"]:
            lane_key = f"{direction.capitalize()} {lane}"
            initial_count = int(def_lane_tracking[lane_key]["initial"] / 4)
            for _ in range(initial_count):
                def_vehicles.append(spawn_new_vehicle(direction, lane, 600, 600, colors))
    
    # Initialize simulation state
    cycle_count = 0
    keep_running = True
    
    # Track vehicle spawning
    opt_last_spawn = time.time()
    def_last_spawn = time.time()
    
    # Track vehicle stats
    opt_total_cleared = 0
    def_total_cleared = 0
    opt_cycle_history = []
    def_cycle_history = []
    
    # Create progress bars
    opt_progress = st.progress(0)
    def_progress = st.progress(0)
    
    # Run simulations side by side
    while keep_running and cycle_count < 5:  # Maximum 5 cycles
        cycle_count += 1
        opt_cycle_info.info(f"Cycle {cycle_count} - Optimized")
        def_cycle_info.info(f"Cycle {cycle_count} - Default")
        
        # Initialize cleared vehicles for this cycle
        opt_cleared_this_cycle = {lane: 0 for lane in opt_lane_tracking}
        def_cleared_this_cycle = {lane: 0 for lane in def_lane_tracking}
        
        # Cycle timing
        opt_cycle_time = sum(optimized_timings)
        def_cycle_time = sum(default_timings)
        opt_cycle_start = time.time()
        def_cycle_start = time.time()
        
        # Maximum cycle duration (use the longer of the two)
        max_cycle_time = max(opt_cycle_time, def_cycle_time)
        
        # Track current traffic lights
        opt_light_idx = 0
        def_light_idx = 0
        opt_light_id = 1
        def_light_id = 1
        opt_light_start = time.time()
        def_light_start = time.time()
        
        # Run both simulations for one cycle
        cycle_end_time = time.time() + max_cycle_time
        while time.time() < cycle_end_time:
            current_time = time.time()
            
            # Check if it's time to change optimized light
            if current_time - opt_light_start >= optimized_timings[opt_light_idx]:
                opt_light_idx = (opt_light_idx + 1) % len(optimized_timings)
                opt_light_id = opt_light_idx + 1
                opt_light_start = current_time
            
            # Check if it's time to change default light
            if current_time - def_light_start >= default_timings[def_light_idx]:
                def_light_idx = (def_light_idx + 1) % len(default_timings)
                def_light_id = def_light_idx + 1
                def_light_start = current_time
            
            # Spawn vehicles for optimized simulation
            if current_time - opt_last_spawn > 2.0:
                opt_last_spawn = current_time
                for direction in ["horizontal", "vertical"]:
                    for lane in ["incoming", "outgoing"]:
                        lane_key = f"{direction.capitalize()} {lane}"
                        if np.random.random() < lane_traffic_rates[lane_key] * 2:
                            opt_vehicles.append(spawn_new_vehicle(direction, lane, 600, 600, colors))
            
            # Spawn vehicles for default simulation
            if current_time - def_last_spawn > 2.0:
                def_last_spawn = current_time
                for direction in ["horizontal", "vertical"]:
                    for lane in ["incoming", "outgoing"]:
                        lane_key = f"{direction.capitalize()} {lane}"
                        if np.random.random() < lane_traffic_rates[lane_key] * 2:
                            def_vehicles.append(spawn_new_vehicle(direction, lane, 600, 600, colors))
            
            # Move vehicles in optimized simulation
            opt_to_remove = []
            for idx, vehicle in enumerate(opt_vehicles):
                exited = vehicle.move(opt_light_id, 600, 600, opt_vehicles)
                if exited:
                    lane_key = f"{vehicle.direction.capitalize()} {vehicle.lane}"
                    opt_cleared_this_cycle[lane_key] += 1
                    opt_total_cleared += 1
                    opt_to_remove.append(idx)
            
            # Remove exited vehicles from optimized simulation
            for idx in sorted(opt_to_remove, reverse=True):
                opt_vehicles.pop(idx)
            
            # Move vehicles in default simulation
            def_to_remove = []
            for idx, vehicle in enumerate(def_vehicles):
                exited = vehicle.move(def_light_id, 600, 600, def_vehicles)
                if exited:
                    lane_key = f"{vehicle.direction.capitalize()} {vehicle.lane}"
                    def_cleared_this_cycle[lane_key] += 1
                    def_total_cleared += 1
                    def_to_remove.append(idx)
            
            # Remove exited vehicles from default simulation
            for idx in sorted(def_to_remove, reverse=True):
                def_vehicles.pop(idx)
            
            # Update displays
            
            # Optimized simulation display update
            opt_counts = count_vehicles_by_lane(opt_vehicles)
            opt_table_data = []
            for lane_key in opt_lane_tracking:
                opt_table_data.append({
                    "Lane": lane_key,
                    "Initial": opt_lane_tracking[lane_key]["initial"],
                    "Current": opt_counts.get(lane_key, 0),
                    "Cleared": opt_cleared_this_cycle[lane_key],
                    "Duration": optimized_timings[list(opt_lane_tracking.keys()).index(lane_key)]
                })
            
            # Add total row for optimized
            opt_table_data.append({
                "Lane": "TOTAL",
                "Initial": sum(data["initial"] for data in opt_lane_tracking.values()),
                "Current": sum(opt_counts.values()),
                "Cleared": sum(opt_cleared_this_cycle.values()),
                "Duration": sum(optimized_timings)
            })
            
            # Default simulation display update
            def_counts = count_vehicles_by_lane(def_vehicles)
            def_table_data = []
            for lane_key in def_lane_tracking:
                def_table_data.append({
                    "Lane": lane_key,
                    "Initial": def_lane_tracking[lane_key]["initial"],
                    "Current": def_counts.get(lane_key, 0),
                    "Cleared": def_cleared_this_cycle[lane_key],
                    "Duration": default_timings[list(def_lane_tracking.keys()).index(lane_key)]
                })
            
            # Add total row for default
            def_table_data.append({
                "Lane": "TOTAL",
                "Initial": sum(data["initial"] for data in def_lane_tracking.values()),
                "Current": sum(def_counts.values()),
                "Cleared": sum(def_cleared_this_cycle.values()),
                "Duration": sum(default_timings)
            })
            
            # Update the table displays
            opt_count_placeholder.dataframe(pd.DataFrame(opt_table_data), use_container_width=True, hide_index=True)
            def_count_placeholder.dataframe(pd.DataFrame(def_table_data), use_container_width=True, hide_index=True)
            
            # Draw the traffic junctions
            opt_img = draw_traffic_junction(active_light=opt_light_id, vehicles=opt_vehicles)
            def_img = draw_traffic_junction(active_light=def_light_id, vehicles=def_vehicles)
            
            # Update the simulation displays
            opt_sim_placeholder.image(opt_img, use_container_width=True)
            def_sim_placeholder.image(def_img, use_container_width=True)
            
            # Update progress bars
            opt_elapsed = current_time - opt_cycle_start
            def_elapsed = current_time - def_cycle_start
            opt_progress.progress(min(1.0, opt_elapsed / opt_cycle_time))
            def_progress.progress(min(1.0, def_elapsed / def_cycle_time))
            
            # Brief pause to control frame rate
            time.sleep(0.1)
        
        # Collect cycle history for both simulations
        opt_cycle_history.append({
            "Cycle": cycle_count,
            "Mode": "Optimized",
            "Cleared": sum(opt_cleared_this_cycle.values()),
            "Total Cleared": opt_total_cleared,
            "Vehicles": len(opt_vehicles),
            "Efficiency": opt_total_cleared / (cycle_count * opt_cycle_time)
        })
        
        def_cycle_history.append({
            "Cycle": cycle_count,
            "Mode": "Default 10s",
            "Cleared": sum(def_cleared_this_cycle.values()),
            "Total Cleared": def_total_cleared,
            "Vehicles": len(def_vehicles),
            "Efficiency": def_total_cleared / (cycle_count * def_cycle_time)
        })
        
        # Display cycle histories
        opt_cycle_summary.dataframe(pd.DataFrame(opt_cycle_history), use_container_width=True, hide_index=True)
        def_cycle_summary.dataframe(pd.DataFrame(def_cycle_history), use_container_width=True, hide_index=True)
    
    # Final comparison
    st.subheader("Simulation Comparison")
    
    # Calculate efficiencies
    opt_efficiency = opt_total_cleared / (cycle_count * opt_cycle_time)
    def_efficiency = def_total_cleared / (cycle_count * def_cycle_time)
    efficiency_diff = (opt_efficiency - def_efficiency) / def_efficiency * 100 if def_efficiency > 0 else 0
    
    comparison_data = {
        "Metric": ["Total Cycles", "Total Cleared", "Vehicles Remaining", "Efficiency (veh/sec)", "Improvement"],
        "AI-Optimized": [cycle_count, opt_total_cleared, len(opt_vehicles), f"{opt_efficiency:.4f}", f"{efficiency_diff:.1f}%"],
        "Default 10s": [cycle_count, def_total_cleared, len(def_vehicles), f"{def_efficiency:.4f}", "baseline"]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    # Highlight the winner
    if opt_total_cleared > def_total_cleared:
        st.success(f"AI-Optimized timing cleared {opt_total_cleared - def_total_cleared} more vehicles ({efficiency_diff:.1f}% more efficient)")
    elif def_total_cleared > opt_total_cleared:
        st.success(f"Default timing cleared {def_total_cleared - opt_total_cleared} more vehicles ({-efficiency_diff:.1f}% more efficient)")
    else:
        st.info("Both simulations cleared the same number of vehicles")

if __name__ == "__main__":
    main()
