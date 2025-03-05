import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
import concurrent.futures

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH

# Constants for vehicle detection
CONF_THRES = 0.25  # Confidence threshold
IOU_THRES = 0.45  # NMS IoU threshold
VEHICLE_CLASSES = [
    2,
    3,
    5,
    7,
]  # COCO class indices for vehicles (car, motorcycle, bus, truck)
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Add new constants for tracking
MAX_TRACKING_DISTANCE = 100  # Maximum pixel distance for tracking association
MIN_DETECTION_CONFIDENCE = 0.4  # Minimum confidence for counting
DETECTION_ZONE_HEIGHT = 0.1  # Height of detection zone as fraction of frame height


def download_model():
    """Download YOLOv5 model if not already downloaded"""
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    try:
        print("Loading YOLOv5 model...")
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model via torch.hub: {e}")
        print("Trying alternative method...")

        # If torch.hub fails, try downloading directly
        if not os.path.exists("yolov5"):
            print("Downloading YOLOv5 repository...")
            os.system("git clone https://github.com/ultralytics/yolov5.git")
            os.system("pip install -r yolov5/requirements.txt")

        # Import the model from the downloaded repository
        sys.path.append("yolov5")
        from yolov5.models.experimental import attempt_load

        weights_path = "yolov5/yolov5s.pt"
        if not os.path.exists(weights_path):
            print("Downloading model weights...")
            os.system(
                f"curl -L -o {weights_path} https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt"
            )

        model = attempt_load(weights_path)
        print("Model loaded successfully via alternative method")
        return model


def create_video_writer(video_path, output_path):
    """Create a video writer based on input video properties"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Try different codecs depending on the platform
    codecs = ["mp4v", "avc1", "XVID"]
    writer = None

    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            temp_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Verify the writer is initialized properly
            if temp_writer.isOpened():
                writer = temp_writer
                print(f"Using codec: {codec}")
                break
            else:
                temp_writer.release()
        except Exception as e:
            print(f"Failed with codec {codec}: {e}")

    if writer is None:
        print(
            "Warning: Could not create video writer with any codec. Output video will not be saved."
        )

    return cap, writer, width, height


def draw_detection_results(
    frame,
    detections,
    vehicle_count,
    class_counts,
    counting_zones=None,
    tracked_paths=None,
):
    """Draw detection results on the frame"""
    height, width = frame.shape[:2]

    # Draw detection zones if provided
    if counting_zones:
        for zone in counting_zones:
            y = int(height * zone["position"])
            direction = "↑" if zone["direction"] == "up" else "↓"
            cv2.line(frame, (0, y), (width, y), zone["color"], 2)
            cv2.putText(
                frame,
                f"Count {direction}: {zone['count']}",
                (width - 200, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                zone["color"],
                2,
            )
    else:
        # Draw default detection line at 60% of the frame height
        line_y = int(height * 0.6)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 2)

    # Remove the tracking path visualization
    # We no longer draw the lines from vehicles to reduce visual clutter

    # Draw vehicle count
    cv2.putText(
        frame,
        f"Total Count: {vehicle_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # Draw class counts
    y_pos = 70
    for class_name, count in class_counts.items():
        if count > 0:
            cv2.putText(
                frame,
                f"{class_name}: {count}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            y_pos += 30

    # Draw detection boxes
    for det in detections:
        if "box" not in det:
            continue

        xyxy, conf, cls_id = det["box"], det["conf"], det["class_id"]
        x1, y1, x2, y2 = map(int, xyxy)

        # Get class name
        class_name = CLASS_NAMES.get(cls_id, "vehicle")

        # Pick color based on class
        if cls_id == 2:  # car
            color = (0, 255, 0)  # green
        elif cls_id == 3:  # motorcycle
            color = (255, 0, 0)  # blue
        elif cls_id == 5:  # bus
            color = (0, 0, 255)  # red
        elif cls_id == 7:  # truck
            color = (0, 165, 255)  # orange
        else:
            color = (255, 255, 0)  # cyan

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw vehicle ID but smaller and less prominent
        if "id" in det:
            id_text = f"{det['id']}"  # Simplified ID display
            cv2.putText(
                frame, id_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

        # Draw label
        label = f"{class_name} {conf:.2f}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return frame


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    # Box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoU
    iou = intersection / (box1_area + box2_area - intersection + 1e-7)
    return iou


def detect_and_track_vehicles(model, video_path, output_path=None, show_video=True):
    """Detect and track vehicles in a video using YOLOv5"""
    # Set up model
    model.classes = VEHICLE_CLASSES  # Filter for vehicle classes only

    # Create output directory if needed
    if output_path and os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Fix output video path if needed
    if output_path:
        # Ensure the extension is .avi which has better compatibility
        if output_path.endswith(".mp4"):
            output_path = output_path.replace(".mp4", ".avi")

    # Open video and create writer
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0, {}

    # Get video dimensions and FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create writer if needed
    writer = None
    if output_path:
        try:
            # Try XVID codec which is widely compatible
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                print(
                    f"Failed to open video writer with XVID codec. Trying alternative..."
                )
                writer.release()

                # Try MJPG codec as an alternative
                output_path = output_path.replace(".avi", "_mjpg.avi")
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                if not writer.isOpened():
                    print("Failed to create video writer. No output will be saved.")
                    writer = None
        except Exception as e:
            print(f"Error creating video writer: {e}")
            writer = None

    # Initialize vehicle counting
    vehicle_count = 0
    class_counts = {class_name: 0 for class_name in CLASS_NAMES.values()}

    # Define counting zones
    counting_zones = [
        {
            "position": 0.6,  # 60% from the top
            "direction": "down",
            "color": (0, 255, 0),  # Green
            "count": 0,
            "already_counted": set(),
        },
        {
            "position": 0.4,  # 40% from the top
            "direction": "up",
            "color": (0, 0, 255),  # Red
            "count": 0,
            "already_counted": set(),
        },
    ]

    # For tracking
    tracked_vehicles = {}  # Dictionary to track vehicles
    next_id = 0  # Next vehicle ID to assign
    tracked_paths = {}  # Store paths for visualization
    max_path_length = 30  # Maximum points to keep in path

    # Set up progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(
        total=total_frames, desc=f"Processing {os.path.basename(video_path)}"
    )

    frame_count = 0
    start_time = time.time()

    # Flag to check if we should stop processing
    should_stop = False

    # Create a unique window name based on video file name
    window_name = f"Vehicle Detection - {os.path.basename(video_path)}"

    try:
        # Process video frames
        while cap.isOpened() and not should_stop:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress_bar.update(1)

            # Run inference
            results = model(frame)

            # Get current detections
            current_detections = []
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                if conf >= CONF_THRES:
                    cls_id = int(cls)
                    class_name = CLASS_NAMES.get(cls_id, "vehicle")

                    x1, y1, x2, y2 = map(int, xyxy)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    current_detections.append(
                        {
                            "box": xyxy,
                            "center": (center_x, center_y),
                            "conf": conf,
                            "class_id": cls_id,
                            "class_name": class_name,
                            "matched": False,
                        }
                    )

            # Match detections with tracked vehicles
            # ...existing code for matching and updating tracks...

            detections_for_display = []

            # First, update existing tracks by matching them with new detections
            for vehicle_id, vehicle_data in list(tracked_vehicles.items()):
                # ...existing tracking code...
                best_match = None
                best_score = -1

                prev_box = vehicle_data["box"]
                prev_center = vehicle_data["center"]
                missed_frames = vehicle_data["missed_frames"]

                # Find the best match for this vehicle
                for det in current_detections:
                    if det["matched"]:
                        continue

                    # Calculate IoU between boxes
                    iou = calculate_iou(prev_box, det["box"])

                    # Calculate center distance
                    center_dist = np.sqrt(
                        (prev_center[0] - det["center"][0]) ** 2
                        + (prev_center[1] - det["center"][1]) ** 2
                    )

                    # Skip if too far away
                    if center_dist > MAX_TRACKING_DISTANCE:
                        continue

                    # Normalize distance score (closer = higher score)
                    dist_score = 1 - min(center_dist / MAX_TRACKING_DISTANCE, 1.0)

                    # Combined score (weighted average of IoU and distance)
                    score = iou * 0.7 + dist_score * 0.3

                    if score > best_score and score > 0.3:  # Minimum score threshold
                        best_score = score
                        best_match = det

                # Update tracking based on match
                # ...existing update code...
                if best_match is not None:
                    best_match["matched"] = True

                    # Update vehicle data
                    tracked_vehicles[vehicle_id].update(
                        {
                            "center": best_match["center"],
                            "box": best_match["box"],
                            "conf": best_match["conf"],
                            "missed_frames": 0,
                            "last_seen": frame_count,
                        }
                    )

                    # Update tracking path
                    if vehicle_id not in tracked_paths:
                        tracked_paths[vehicle_id] = []
                    tracked_paths[vehicle_id].append(best_match["center"])
                    if len(tracked_paths[vehicle_id]) > max_path_length:
                        tracked_paths[vehicle_id] = tracked_paths[vehicle_id][
                            -max_path_length:
                        ]

                    # Check for crossing detection zones
                    center_y = best_match["center"][1]
                    for zone in counting_zones:
                        zone_y = int(height * zone["position"])
                        prev_center_y = prev_center[1]

                        # Check if vehicle crossed the line in the correct direction
                        crossed_down = (
                            prev_center_y < zone_y <= center_y
                            and zone["direction"] == "down"
                        )
                        crossed_up = (
                            prev_center_y > zone_y >= center_y
                            and zone["direction"] == "up"
                        )

                        if (crossed_down or crossed_up) and vehicle_id not in zone[
                            "already_counted"
                        ]:
                            if best_match["conf"] >= MIN_DETECTION_CONFIDENCE:
                                zone["count"] += 1
                                zone["already_counted"].add(vehicle_id)
                                vehicle_count += 1
                                class_counts[best_match["class_name"]] += 1
                                print(
                                    f"Vehicle {vehicle_id} ({best_match['class_name']}) crossed {zone['direction']} line"
                                )

                    # Add to detections for display
                    display_det = best_match.copy()
                    display_det["id"] = vehicle_id
                    detections_for_display.append(display_det)
                else:
                    # Increment missed frames counter
                    tracked_vehicles[vehicle_id]["missed_frames"] += 1

                    # Remove the track if it's been missing for too long
                    if tracked_vehicles[vehicle_id]["missed_frames"] > 10:
                        del tracked_vehicles[vehicle_id]
                        if vehicle_id in tracked_paths:
                            del tracked_paths[vehicle_id]
                    else:
                        # Add the last known position for display
                        detections_for_display.append(
                            {
                                "box": tracked_vehicles[vehicle_id]["box"],
                                "conf": tracked_vehicles[vehicle_id]["conf"],
                                "class_id": tracked_vehicles[vehicle_id]["class_id"],
                                "class_name": tracked_vehicles[vehicle_id][
                                    "class_name"
                                ],
                                "id": vehicle_id,
                                "center": tracked_vehicles[vehicle_id]["center"],
                            }
                        )

            # Create new tracks for unmatched detections
            for det in current_detections:
                if not det["matched"] and det["conf"] >= MIN_DETECTION_CONFIDENCE:
                    vehicle_id = next_id
                    next_id += 1

                    tracked_vehicles[vehicle_id] = {
                        "box": det["box"],
                        "center": det["center"],
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "conf": det["conf"],
                        "first_seen": frame_count,
                        "last_seen": frame_count,
                        "missed_frames": 0,
                    }

                    # Start a new tracking path
                    tracked_paths[vehicle_id] = [det["center"]]

                    # Add to detections for display with the assigned ID
                    display_det = det.copy()
                    display_det["id"] = vehicle_id
                    detections_for_display.append(display_det)

            # Draw results on frame
            frame = draw_detection_results(
                frame,
                detections_for_display,
                vehicle_count,
                class_counts,
                counting_zones=counting_zones,
                tracked_paths=None,  # Don't pass tracked paths to remove lines
            )

            # Write frame to output video
            if writer is not None:
                try:
                    # Only write if frame has valid dimensions
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        # Ensure frame is the right size
                        if frame.shape[1] != width or frame.shape[0] != height:
                            frame = cv2.resize(frame, (width, height))

                        writer.write(frame)
                    else:
                        print(
                            f"Warning: Invalid frame dimensions at frame {frame_count}"
                        )
                except Exception as e:
                    print(f"Error writing frame {frame_count}: {e}")
                    if writer is not None:
                        writer.release()
                        writer = None

            # Display frame - always show if requested
            if show_video:
                # Resize frame for display to make it smaller
                display_frame = cv2.resize(frame, (int(width/2), int(height/2)))
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):  # Press 'q' to quit
                    should_stop = True
                    break

            # Display progress (every 30 frames)
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                progress_bar.set_postfix(
                    {"FPS": f"{fps:.1f}", "Vehicles": vehicle_count}
                )

    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        progress_bar.close()
        cap.release()

        if writer is not None:
            writer.release()

            # Verify the output file was created correctly
            if os.path.exists(output_path):
                filesize = os.path.getsize(output_path)
                if filesize < 10000:  # Less than 10KB is suspicious
                    print(f"Warning: Output file seems too small ({filesize} bytes)")
            else:
                print(f"Warning: Output file was not created at {output_path}")

        if show_video:
            cv2.destroyWindow(window_name)

    # Print final results
    print(f"\n--- Results for {os.path.basename(video_path)} ---")
    print(f"Total vehicles detected: {vehicle_count}")
    for zone in counting_zones:
        print(
            f"Zone at {int(zone['position'] * 100)}%: {zone['count']} vehicles going {zone['direction']}"
        )
    print("Vehicle class breakdown:")
    for class_name, count in class_counts.items():
        if count > 0:
            print(f"  {class_name}: {count}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")

    return vehicle_count, class_counts


def process_single_video(args):
    """Process a single video - worker function for multiprocessing"""
    video_path, output_path, show_video = args
    
    # Each process loads its own model
    print(f"\nProcess {os.getpid()}: Loading model for {os.path.basename(video_path)}...")
    model = download_model()
    
    print(f"\nProcessing video: {os.path.basename(video_path)}")
    try:
        vehicle_count, class_counts = detect_and_track_vehicles(
            model, video_path, output_path, show_video
        )
        return video_path, vehicle_count, class_counts
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return video_path, 0, {}

def process_videos(video_paths, output_dir=None, show_video=True):
    """Process multiple videos in parallel using concurrent.futures"""
    # Don't preload the model in the main process
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Prepare arguments for each video (without the model)
    tasks = []
    for video_path in video_paths:
        # Create output path if output directory is specified
        if output_dir:
            video_name = os.path.basename(video_path)
            video_name_base = os.path.splitext(video_name)[0]
            output_path = os.path.join(output_dir, f"output_{video_name_base}.avi")
            counter = 1
            while os.path.exists(output_path):
                output_path = os.path.join(
                    output_dir, f"output_{counter}_{video_name_base}.avi"
                )
                counter += 1
        else:
            output_path = None
            
        # Pass all parameters except model
        # Always pass show_video=True to see the processing
        tasks.append((video_path, output_path, True))

    # Process videos in parallel using ProcessPoolExecutor
    results = {}
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(video_paths)) as executor:
            futures = {executor.submit(process_single_video, task): task for task in tasks}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    video_path, vehicle_count, class_counts = future.result()
                    results[video_path] = {"count": vehicle_count, "classes": class_counts}
                except Exception as exc:
                    print(f"Video processing generated an exception: {exc}")
    except KeyboardInterrupt:
        print("\nDetected keyboard interrupt. Stopping processing gracefully...")
        # Let the finally block handle cleanup

    # Print summary of all videos
    print("\n===== SUMMARY OF ALL VIDEOS =====")
    total_count = 0
    total_by_class = {}

    # ...existing code for summary printing...
    for video_path, data in results.items():
        video_name = os.path.basename(video_path)
        print(f"{video_name}: {data['count']} vehicles")

        # Add to totals
        total_count += data["count"]
        for cls, count in data["classes"].items():
            if cls not in total_by_class:
                total_by_class[cls] = 0
            total_by_class[cls] += count

    print(f"\nTotal vehicles across all videos: {total_count}")
    print("Breakdown by class:")
    for cls, count in total_by_class.items():
        print(f"  {cls}: {count}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect vehicles in videos using YOLOv5"
    )
    parser.add_argument(
        "--videos", nargs="+", default=[], help="Video files to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_videos",
        help="Output directory for processed videos",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display during processing",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process videos sequentially instead of in parallel",
    )

    args = parser.parse_args()

    # Default videos if none specified
    if not args.videos:
        video_dir = os.path.join(ROOT, "videos")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
            print(f"Created video directory: {video_dir}")
            print(
                "Please place your videos in this directory and run the script again."
            )
            print(
                "Default video names expected: video1.mp4, video2.mp4, video3.mp4, video4.mp4"
            )
            return

        # Find all video files in the directory
        video_files = []
        for file in os.listdir(video_dir):
            lower_file = file.lower()
            if lower_file.endswith(('.mp4', '.avi', '.mov')):
                full_path = os.path.join(video_dir, file)
                # Validate the video file can be opened
                try:
                    cap = cv2.VideoCapture(full_path)
                    if cap.isOpened():
                        video_files.append(full_path)
                        cap.release()
                        print(f"Found valid video: {file}")
                    else:
                        print(f"Warning: Could not open {file} as a video file")
                except Exception as e:
                    print(f"Error checking video file {file}: {e}")

        if not video_files:
            print(
                "No valid video files found. Please place videos in the 'videos' directory."
            )
            return
        
        args.videos = sorted(video_files)
    
    # Verify all specified videos exist and can be opened
    valid_videos = []
    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            continue
            
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                valid_videos.append(video_path)
                cap.release()
                print(f"Validated video: {os.path.basename(video_path)}")
            else:
                print(f"Warning: Could not open {video_path} as a video file")
        except Exception as e:
            print(f"Error validating {video_path}: {e}")
    
    if not valid_videos:
        print("No valid videos to process.")
        return
        
    print(f"\nProcessing {len(valid_videos)} video(s): {[os.path.basename(v) for v in valid_videos]}")
    
    # Process videos
    output_dir = os.path.join(ROOT, args.output)
    
    # Add option to process sequentially if parallel processing fails
    if args.sequential:
        results = {}
        model = download_model()
        for video_path in valid_videos:
            output_path = os.path.join(output_dir, f"output_{os.path.basename(video_path)}")
            try:
                print(f"\nProcessing {os.path.basename(video_path)} sequentially...")
                vehicle_count, class_counts = detect_and_track_vehicles(
                    model, video_path, output_path, not args.no_display
                )
                results[video_path] = {"count": vehicle_count, "classes": class_counts}
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
    else:
        try:
            results = process_videos(valid_videos, output_dir, not args.no_display)
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            print("Falling back to sequential processing...")
            
            results = {}
            model = download_model()
            for video_path in valid_videos:
                output_path = os.path.join(output_dir, f"output_{os.path.basename(video_path)}")
                try:
                    print(f"\nProcessing {os.path.basename(video_path)} sequentially...")
                    vehicle_count, class_counts = detect_and_track_vehicles(
                        model, video_path, output_path, not args.no_display
                    )
                    results[video_path] = {"count": vehicle_count, "classes": class_counts}
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    main()
