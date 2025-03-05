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

    def move(self, active_light, width, height, all_vehicles):
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
        
        # Scale down the count to a manageable number for display
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
    
    # Create a placeholder for the simulation
    simulation_placeholder = st.empty()
    
    # Start button for the simulation
    if st.button("Start Traffic Light Simulation"):
        # Initialize vehicles based on detected counts
        video_counts = {os.path.basename(path): data['total_count'] for path, data in results.items()}
        vehicles = initialize_vehicles(video_counts, 600, 600)
        
        # Run through the traffic light sequence
        cycle_time = sum(timings)
        st.write(f"Total cycle time: {cycle_time} seconds")
        
        # Show a progress bar for the full cycle
        cycle_progress = st.progress(0)
        
        # Start time for the cycle
        start_time = time.time()
        
        # Run one full cycle
        for i in range(len(timings)):
            light_id = i + 1
            duration = timings[i]
            lane = ["Horizontal incoming", "Horizontal outgoing", "Vertical incoming", "Vertical outgoing"][i]
            
            st.write(f"Green light for {lane} - Duration: {duration}s")
            
            # Draw the junction with the current active light
            light_start = time.time()
            
            # Run for the duration of this light
            while time.time() - light_start < duration:
                # Move vehicles with collision detection
                for vehicle in vehicles:
                    vehicle.move(light_id, 600, 600, vehicles)
                
                # Draw the scene
                junction_image = draw_traffic_junction(active_light=light_id, vehicles=vehicles)
                # Replace deprecated use_column_width with use_container_width
                simulation_placeholder.image(junction_image, use_container_width=True)
                
                # Update progress bar
                current_time = time.time() - start_time
                progress = min(1.0, current_time / cycle_time)
                cycle_progress.progress(progress)
                
                # Sleep to control the frame rate
                time.sleep(0.1)
    
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

if __name__ == "__main__":
    main()
