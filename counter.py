import cv2
import numpy as np
from collections import defaultdict
import time
import os
import multiprocessing
from multiprocessing import Manager, Process

class VehicleCounter:
    def __init__(self, detection_line_position=0.5):
        # Initialize the background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)
        
        # Initialize vehicle count and tracking
        self.vehicle_count = 0
        self.tracked_vehicles = defaultdict(int)
        self.tracked_positions = {}
        self.next_vehicle_id = 0
        
        # Detection line position (percentage of frame height from top)
        self.detection_line_position = detection_line_position
        self.detection_line_y = None
        
        # Minimum contour area to be considered a vehicle (will be set based on frame size)
        self.min_contour_area = None
        
    def process_video(self, video_path, output_path=None):
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Set detection line and minimum contour area based on frame size
        self.detection_line_y = int(frame_height * self.detection_line_position)
        self.min_contour_area = (frame_width * frame_height) // 400  # Adaptive threshold
        
        # Initialize video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Process each frame
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process frame
            result_frame = self.process_frame(frame)
            
            # Write frame if output is specified
            if writer:
                writer.write(result_frame)
            
            # Display progress
            if frame_count % 30 == 0:  # Update every 30 frames
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"\rProcessing frame {frame_count} | FPS: {fps:.2f} | Vehicles counted: {self.vehicle_count}", end="")
            
            # Display frame (comment out for faster processing)
            cv2.imshow('Vehicle Counter', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print final results
        print(f"\n\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total vehicles counted: {self.vehicle_count}")
        print(f"Processing time: {time.time() - start_time:.2f} seconds")
        
    def process_frame(self, frame):
        # Create a copy of the frame
        result_frame = frame.copy()
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw detection line
        cv2.line(result_frame, (0, self.detection_line_y), 
                 (result_frame.shape[1], self.detection_line_y), (0, 255, 0), 2)
        
        # Process each contour
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + h // 2
            
            # Check if the vehicle center crossed the detection line
            vehicle_id = None
            min_distance = float('inf')
            
            # Try to match with existing vehicles
            for vid, pos in self.tracked_positions.items():
                distance = abs(pos[0] - x) + abs(pos[1] - y)
                if distance < min_distance:
                    min_distance = distance
                    if distance < 50:  # Maximum allowed distance for matching
                        vehicle_id = vid
            
            if vehicle_id is None:
                vehicle_id = self.next_vehicle_id
                self.next_vehicle_id += 1
            
            # Update tracked positions
            self.tracked_positions[vehicle_id] = (x, y)
            
            # Check if vehicle crossed the line
            if self.tracked_vehicles[vehicle_id] == 0:
                if center_y < self.detection_line_y and y + h > self.detection_line_y:
                    self.vehicle_count += 1
                    self.tracked_vehicles[vehicle_id] = 1
            
            # Draw bounding box and ID
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_frame, f'Vehicle {vehicle_id}', 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw vehicle count
        cv2.putText(result_frame, f'Vehicle Count: {self.vehicle_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result_frame

def process_video_path(video_path, output_path, counts_dict):
    counter = VehicleCounter(detection_line_position=0.6)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counter.detection_line_y = int(frame_height * counter.detection_line_position)
    counter.min_contour_area = (frame_width * frame_height) // 400

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = counter.process_frame(frame)
        if writer:
            writer.write(result_frame)
        counts_dict[video_path] = counter.vehicle_count
        # Removed cv2.imshow and cv2.waitKey to prevent blocking

    cap.release()
    if writer:
        writer.release()
    # Removed cv2.destroyAllWindows()

def main():
    # Create an instance of VehicleCounter
    counter = VehicleCounter(detection_line_position=0.6)  # Line at 60% from top of frame
    
    # List of paths to your video files
    video_paths = [
        "videos/video1.mp4",
        "videos/video2.mp4",
        "videos/video3.mp4",
        "videos/video4.mp4"
    ]
    
    # Optional: Corresponding output paths for each video
    output_paths = [
        "videos/outputVideo1.mp4",
        "videos/outputVideo2.mp4",
        "videos/outputVideo3.mp4",
        "videos/outputVideo4.mp4"
    ]
    
    # Dictionary to store vehicle counts for each video
    vehicle_counts = {}
    
    manager = Manager()
    counts_dict = manager.dict()
    processes = []
    for idx, video_path in enumerate(video_paths):
        output_path = output_paths[idx] if output_paths else None
        p = Process(target=process_video_path, args=(video_path, output_path, counts_dict))
        p.start()
        processes.append(p)
    
    # Continually check counts while processes are alive
    try:
        while any(p.is_alive() for p in processes):
            print("\rCurrent vehicle counts:", dict(counts_dict), end="")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    for p in processes:
        p.join()

    for video_path, count in counts_dict.items():
        vehicle_counts[video_path] = count
        print(f"Total vehicles counted in {video_path}: {count}")
    
    # Print summary
    print("\nVehicle Counts Summary:")
    for video, count in vehicle_counts.items():
        print(f"{video}: {count} vehicles")
    
if __name__ == "__main__":
    main()