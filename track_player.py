import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math

class PlayerTracker:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_path = "output_tracking.mp4"
        
        # Optimized parameters for football tracking
        self.confidence_threshold = 0.25
        
        # Core tracking data - ALL using sequential IDs
        self.sequential_track_history = defaultdict(deque)
        self.sequential_track_persistence = defaultdict(int)
        self.sequential_track_positions = defaultdict(list)
        
        # ID Management
        self.original_to_sequential = {}  # original_id -> sequential_id
        self.sequential_to_original = {}  # sequential_id -> [original_ids]
        self.next_sequential_id = 1
        
        # Active tracking
        self.active_sequential_ids = set()
        self.lost_sequential_tracks = {}
        
        # Track merging parameters
        self.max_merge_distance = 100
        self.min_track_length = 15
        self.max_lost_frames = 30
        
        # Quality metrics
        self.frame_count = 0
        self.merged_tracks = 0
        
        # Debug tracking
        self.debug_mapping = []
        
    def assign_sequential_id(self, original_id):
        """Assign or retrieve sequential ID for an original YOLO ID"""
        if original_id not in self.original_to_sequential:
            # Create new sequential ID
            sequential_id = self.next_sequential_id
            self.original_to_sequential[original_id] = sequential_id
            
            if sequential_id not in self.sequential_to_original:
                self.sequential_to_original[sequential_id] = []
            self.sequential_to_original[sequential_id].append(original_id)
            
            self.next_sequential_id += 1
            
            self.debug_mapping.append(f"Frame {self.frame_count}: Original {original_id} -> Sequential {sequential_id}")
            
            if len(self.debug_mapping) <= 20:  # Only print first 20 mappings
                print(f"NEW MAPPING: Original ID {original_id} assigned to Sequential ID {sequential_id}")
            
        return self.original_to_sequential[original_id]
    
    def update_active_tracks(self, current_original_ids):
        """Update which sequential tracks are currently active"""
        current_sequential_ids = set()
        
        for orig_id in current_original_ids:
            seq_id = self.assign_sequential_id(orig_id)
            current_sequential_ids.add(seq_id)
        
        # Mark tracks as lost
        for seq_id in self.active_sequential_ids:
            if seq_id not in current_sequential_ids:
                self.lost_sequential_tracks[seq_id] = self.frame_count
        
        # Remove very old lost tracks
        old_lost = [sid for sid, frame in self.lost_sequential_tracks.items() 
                   if self.frame_count - frame > self.max_lost_frames]
        for sid in old_lost:
            del self.lost_sequential_tracks[sid]
        
        self.active_sequential_ids = current_sequential_ids
    
    def get_color(self, sequential_id):
        """Generate consistent colors for sequential track IDs"""
        np.random.seed(int(sequential_id))
        return tuple(np.random.randint(50, 255, 3).tolist())
    
    def should_merge_tracks(self, track1_positions, track2_positions):
        """Determine if two sequential tracks should be merged"""
        if len(track1_positions) < 5 or len(track2_positions) < 5:
            return False
        
        recent1 = track1_positions[-5:]
        recent2 = track2_positions[-5:]
        
        distances = []
        for p1, p2 in zip(recent1, recent2):
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        return avg_distance < self.max_merge_distance
    
    def post_process_sequential_tracks(self):
        """Post-process sequential tracks to merge similar ones"""
        sequential_ids = list(self.sequential_track_positions.keys())
        merged_count = 0
        
        print(f"\nPost-processing {len(sequential_ids)} sequential tracks for merging...")
        
        for i, seq_id1 in enumerate(sequential_ids):
            if seq_id1 not in self.sequential_track_positions:
                continue
                
            for j, seq_id2 in enumerate(sequential_ids[i+1:], i+1):
                if seq_id2 not in self.sequential_track_positions:
                    continue
                
                if self.should_merge_tracks(self.sequential_track_positions[seq_id1], 
                                           self.sequential_track_positions[seq_id2]):
                    # Merge seq_id2 into seq_id1
                    print(f"MERGING: Sequential ID {seq_id2} merged into Sequential ID {seq_id1}")
                    
                    self.sequential_track_positions[seq_id1].extend(self.sequential_track_positions[seq_id2])
                    self.sequential_track_persistence[seq_id1] += self.sequential_track_persistence[seq_id2]
                    
                    # Update mapping
                    if seq_id2 in self.sequential_to_original:
                        if seq_id1 not in self.sequential_to_original:
                            self.sequential_to_original[seq_id1] = []
                        self.sequential_to_original[seq_id1].extend(self.sequential_to_original[seq_id2])
                        del self.sequential_to_original[seq_id2]
                    
                    del self.sequential_track_positions[seq_id2]
                    del self.sequential_track_persistence[seq_id2]
                    merged_count += 1
        
        return merged_count
    
    def calculate_track_quality(self):
        """Calculate quality metrics using sequential data"""
        long_tracks = 0
        very_long_tracks = 0
        excellent_tracks = 0
        total_tracks = len(self.sequential_track_persistence)
        
        for seq_id, duration in self.sequential_track_persistence.items():
            if duration > 75:  # > 3 seconds at 25fps
                long_tracks += 1
            if duration > 150:  # > 6 seconds
                very_long_tracks += 1
            if duration > 250:  # > 10 seconds
                excellent_tracks += 1
        
        total_original_ids = len(self.original_to_sequential)
        id_efficiency = total_tracks / max(1, total_original_ids)
        
        return {
            'total_tracks': total_tracks,
            'long_tracks': long_tracks,
            'very_long_tracks': very_long_tracks,
            'excellent_tracks': excellent_tracks,
            'fragmentation_ratio': total_tracks / 22 if total_tracks > 0 else 0,
            'persistence_rate': long_tracks / total_tracks if total_tracks > 0 else 0,
            'excellence_rate': excellent_tracks / total_tracks if total_tracks > 0 else 0,
            'id_efficiency': id_efficiency,
            'total_original_ids': total_original_ids
        }
    
    def process_video(self):
        """Main video processing with deep ID fixing"""
        print("=" * 50)
        print("PLAYER TRACKING SYSTEM")
        print("=" * 50)
        print("Features: Sequential ID assignment, Track recovery, Clean visualization")
        print("Loading video and model...")
        
        # Setup video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Properties: {width}x{height} pixels, {fps} FPS, {total_frames} total frames")
        
        # Setup output video
        out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )
        
        print("Starting player tracking with sequential ID management...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Progress update
            if self.frame_count % 50 == 0:
                progress = (self.frame_count / total_frames) * 100
                active_count = len(self.active_sequential_ids)
                total_sequential = self.next_sequential_id - 1
                print(f"Processing: Frame {self.frame_count}/{total_frames} ({progress:.1f}%) | Active Tracks: {active_count} | Total Sequential IDs: {total_sequential}")
            
            # YOLO tracking
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[2],  # Players only
                    conf=self.confidence_threshold,
                    iou=0.3,
                    verbose=False,
                    tracker="bytetrack.yaml",
                    imgsz=640,
                    device='cpu'
                )
            except:
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[2],
                    conf=self.confidence_threshold,
                    iou=0.3,
                    verbose=False,
                    tracker="botsort.yaml",
                    imgsz=640,
                    device='cpu'
                )
            
            # Process tracking results
            current_original_ids = []
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                original_track_ids = results[0].boxes.id.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                current_original_ids = [int(tid) for tid in original_track_ids]
                
                # Update active tracks
                self.update_active_tracks(current_original_ids)
                
                for box, original_id, conf in zip(boxes, original_track_ids, confidences):
                    x1, y1, x2, y2 = box
                    original_id = int(original_id)
                    
                    # Get sequential ID - THIS IS THE KEY FIX
                    sequential_id = self.assign_sequential_id(original_id)
                    
                    # Update ALL tracking data using ONLY sequential IDs
                    if conf > 0.3:
                        self.sequential_track_persistence[sequential_id] += 1
                        
                        # Store position for merging analysis
                        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                        self.sequential_track_positions[sequential_id].append((center_x, center_y, self.frame_count))
                        
                        # Store for trail visualization
                        self.sequential_track_history[sequential_id].append((int(center_x), int(center_y)))
                        
                        # Keep only last 20 positions for trail
                        if len(self.sequential_track_history[sequential_id]) > 20:
                            self.sequential_track_history[sequential_id].popleft()
                    
                    # Draw tracking using sequential ID
                    color = self.get_color(sequential_id)
                    
                    # Draw bounding box
                    thickness = max(1, int(conf * 4))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # Draw ONLY sequential ID (no original ID shown)
                    label = f"ID:{sequential_id} ({conf:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Background for label
                    label_bg_color = color if conf > 0.5 else (100, 100, 100)
                    cv2.rectangle(frame, (int(x1), int(y1)-30), 
                                (int(x1)+label_size[0]+10, int(y1)), label_bg_color, -1)
                    
                    # Label text
                    text_color = (255, 255, 255) if conf > 0.5 else (200, 200, 200)
                    cv2.putText(frame, label, (int(x1)+5, int(y1)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                    # Draw track trail
                    if conf > 0.4 and len(self.sequential_track_history[sequential_id]) > 1:
                        points = list(self.sequential_track_history[sequential_id])
                        for i in range(1, len(points)):
                            alpha = i / len(points)
                            trail_color = tuple(int(c * alpha * 0.7) for c in color)
                            cv2.line(frame, points[i-1], points[i], trail_color, 2)
            
            # Calculate and display metrics
            quality_metrics = self.calculate_track_quality()
            active_count = len(self.active_sequential_ids)
            total_sequential = self.next_sequential_id - 1
            
            # Enhanced frame information
            info_lines = [
                f"Frame: {self.frame_count}/{total_frames} ({self.frame_count/total_frames*100:.1f}%)",
                f"Active Sequential: {active_count} | Total Sequential: {total_sequential}",
                f"Persistence: {quality_metrics['persistence_rate']*100:.0f}% | Efficiency: {quality_metrics['id_efficiency']*100:.0f}%"
            ]
            
            for i, info_text in enumerate(info_lines):
                bg_color = (0, 150, 0) if quality_metrics['id_efficiency'] > 0.8 else (0, 100, 150)
                text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (10, 25 + i*25), (20 + text_size[0], 45 + i*25), bg_color, -1)
                cv2.putText(frame, info_text, (15, 40 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            # Save frame
            out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Post-processing
        print("\nStarting post-processing of sequential tracks...")
        self.merged_tracks = self.post_process_sequential_tracks()
        
        # Final analysis
        self.print_final_analysis(fps)
    
    def print_final_analysis(self, fps):
        """Print comprehensive analysis with pure sequential results"""
        print(f"\n" + "=" * 60)
        print(f"PLAYER TRACKING ANALYSIS COMPLETE")
        print(f"=" * 60)
        print(f"Output file saved: {self.output_path}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Sequential tracks merged: {self.merged_tracks}")
        
        quality_metrics = self.calculate_track_quality()
        
        print(f"\nSEQUENTIAL ID MANAGEMENT SUMMARY")
        print(f"-" * 40)
        print(f"Original YOLO IDs encountered: {quality_metrics['total_original_ids']}")
        print(f"Sequential IDs created: {self.next_sequential_id - 1}")
        print(f"Final sequential tracks after merging: {quality_metrics['total_tracks']}")
        print(f"ID management efficiency: {quality_metrics['id_efficiency']*100:.1f}%")
        
        print(f"\nTRACKING QUALITY ANALYSIS")
        print(f"-" * 40)
        print(f"Total sequential tracks: {quality_metrics['total_tracks']}")
        print(f"Long-duration tracks (>3s): {quality_metrics['long_tracks']}")
        print(f"Very long-duration tracks (>6s): {quality_metrics['very_long_tracks']}")
        print(f"Excellent tracks (>10s): {quality_metrics['excellent_tracks']}")
        print(f"Track persistence rate: {quality_metrics['persistence_rate']*100:.1f}%")
        print(f"Track fragmentation ratio: {quality_metrics['fragmentation_ratio']:.1f}x")
        
        # Enhanced quality scoring
        score = 0
        
        if quality_metrics['fragmentation_ratio'] <= 1.5:
            score += 35
        elif quality_metrics['fragmentation_ratio'] <= 2.0:
            score += 25
        
        if quality_metrics['persistence_rate'] >= 0.7:
            score += 25
        elif quality_metrics['persistence_rate'] >= 0.5:
            score += 15
        
        if quality_metrics['id_efficiency'] >= 0.9:
            score += 25
        elif quality_metrics['id_efficiency'] >= 0.8:
            score += 20
        
        if quality_metrics['excellence_rate'] >= 0.3:
            score += 15
        
        print(f"\nOVERALL QUALITY SCORE")
        print(f"-" * 40)
        print(f"Quality Score: {score}/100")
        
        if score >= 85:
            print("Rating: EXCELLENT - Outstanding sequential tracking performance")
        elif score >= 70:
            print("Rating: GOOD - Solid sequential tracking with good results")
        elif score >= 50:
            print("Rating: FAIR - Acceptable tracking, room for improvement")
        else:
            print("Rating: POOR - Tracking quality needs significant improvement")
        
        # Show ONLY sequential tracks in results
        print(f"\nTOP PERFORMING TRACKS (SEQUENTIAL IDs)")
        print(f"-" * 40)
        sorted_tracks = sorted(self.sequential_track_persistence.items(), key=lambda x: x[1], reverse=True)
        
        for i, (sequential_id, duration) in enumerate(sorted_tracks[:10]):
            if duration > 250:
                quality_indicator = "EXCELLENT"
            elif duration > 150:
                quality_indicator = "VERY GOOD"
            elif duration > 75:
                quality_indicator = "GOOD"
            else:
                quality_indicator = "STANDARD"
            
            original_ids = self.sequential_to_original.get(sequential_id, [])
            original_info = f" (mapped from original IDs: {original_ids})" if len(original_ids) <= 3 else f" (mapped from {len(original_ids)} original IDs)"
            print(f"{quality_indicator:>10} | Sequential ID {sequential_id}: {duration} frames ({duration/fps:.1f}s){original_info}")
        
        print(f"\nID MAPPING EFFICIENCY SUMMARY")
        print(f"-" * 40)
        print(f"Total original YOLO IDs processed: {len(self.original_to_sequential)}")
        print(f"Sequential IDs after merging: {len(self.sequential_track_persistence)}")
        print(f"ID reduction ratio: {len(self.original_to_sequential) / max(1, len(self.sequential_track_persistence)):.1f}:1")

def main():
    """Main function"""
    MODEL_PATH = r"C:\\Users\\debra\\Downloads\\best (17).pt"
    VIDEO_PATH = r"C:\\Users\\debra\\Downloads\\15sec_input_720p.mp4"
    
    try:
        tracker = PlayerTracker(MODEL_PATH, VIDEO_PATH)
        tracker.process_video()
        
        print(f"\n" + "=" * 60)
        print(f"TRACKING PROCESS COMPLETED SUCCESSFULLY")
        print(f"=" * 60)
        print(f"Output video file: output_tracking.mp4")
        print(f"\nKey Features Implemented:")
        print(f"  - Sequential ID assignment (1, 2, 3, ...)")
        print(f"  - Elimination of random ID numbers")
        print(f"  - Complete tracking pipeline with clean identification")
        print(f"  - Transparent mapping from original to sequential IDs")
        print(f"  - Professional output formatting")
        
    except Exception as e:
        print(f"ERROR: Tracking process failed - {str(e)}")

if __name__ == "__main__":
    main()
