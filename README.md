# Soccer Player Identification and Tracking System

## 🎯 Project Overview

This project implements an intelligent soccer player identification and tracking system using YOLO (You Only Look Once) deep learning model and advanced computer vision techniques. The system processes soccer match videos to automatically detect, track, and assign consistent sequential IDs to players throughout the video.

## 📋 Table of Contents
- [Concept and Methodology](#concept-and-methodology)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Output Analysis](#output-analysis)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

## 🧠 Concept and Methodology

### The Problem
Traditional object detection models like YOLO assign random IDs to detected objects, which change unpredictably across frames. For sports analysis, we need:
- **Consistent Player Identification**: Same player should have the same ID throughout the video
- **Sequential ID Assignment**: Clean numbering (1, 2, 3...) instead of random numbers
- **Track Continuity**: Maintain tracking even when players are temporarily occluded

### The Solution
This system implements a sophisticated **Sequential ID Management Pipeline** that:

1. **Primary Detection**: Uses YOLO model to detect players in each frame
2. **ID Mapping**: Converts random YOLO IDs to sequential IDs (1, 2, 3...)
3. **Track Persistence**: Maintains player identity across frames using motion prediction
4. **Track Merging**: Combines fragmented tracks of the same player
5. **Quality Assessment**: Evaluates tracking performance with comprehensive metrics

### Core Algorithm Flow
```
Input Video → YOLO Detection → ID Assignment → Track Management → Output Video
     ↓              ↓              ↓              ↓              ↓
  Raw frames → Player boxes → Sequential IDs → Track merging → Labeled video
```

## ✨ Key Features

### 🔢 Sequential ID Management
- Assigns clean, sequential IDs (1, 2, 3...) to players
- Eliminates confusing random YOLO ID numbers
- Maintains consistent player identification throughout the video

### 🎯 Advanced Tracking
- **ByteTrack/BoTSORT Integration**: Uses state-of-the-art tracking algorithms
- **Motion Prediction**: Predicts player positions during occlusions
- **Track Recovery**: Reconnects lost tracks when players reappear

### 🔄 Intelligent Track Merging
- Automatically merges fragmented tracks of the same player
- Uses position analysis and motion patterns for merging decisions
- Reduces track fragmentation significantly

### 📊 Real-time Visualization
- Live bounding boxes with confidence scores
- Motion trails showing player movement paths
- On-screen performance metrics and statistics

### 📈 Performance Analytics
- Track persistence rates and quality metrics
- ID management efficiency analysis
- Comprehensive tracking quality scoring

## 🏗️ Technical Architecture

### Core Components

1. **PlayerTracker Class**: Main tracking engine
   - Video processing pipeline
   - ID management system
   - Track merging algorithms

2. **Detection Engine**: YOLO-based player detection
   - Confidence threshold filtering
   - Class-specific detection (players only)
   - Real-time inference optimization

3. **ID Management System**: 
   - Original-to-Sequential ID mapping
   - Active track monitoring
   - Lost track recovery

4. **Track Analysis Engine**:
   - Position-based track merging
   - Quality metric calculation
   - Performance assessment

### Key Algorithms

#### Sequential ID Assignment
```python
def assign_sequential_id(self, original_id):
    if original_id not in self.original_to_sequential:
        sequential_id = self.next_sequential_id
        self.original_to_sequential[original_id] = sequential_id
        self.next_sequential_id += 1
    return self.original_to_sequential[original_id]
```

#### Track Merging Logic
- Analyzes recent position history of tracks
- Calculates average distance between track paths
- Merges tracks with distance < threshold (100 pixels)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/debraj-m/Soccer-Player-Identification.git
cd Soccer-Player-Identification
```

### Step 2: Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Option 1: Install from requirements.txt (Recommended)
pip install -r requirements.txt

# Option 2: Install manually
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install numpy>=1.21.0

# Verify installation
python -c "import cv2, ultralytics; print('Installation successful!')"
```

### Step 4: Download Pre-trained Model
Place your trained YOLO model file (`best.pt`) in the project directory. This should be a YOLO model trained specifically for soccer player detection.

## 🎮 Usage

### Basic Usage

1. **Prepare Your Files**:
   - Place your input video file in the project directory
   - Ensure your YOLO model (`best.pt`) is available
   - Update file paths in the script if necessary

2. **Run the Tracking System**:
   ```bash
   python track_player.py
   ```

3. **Monitor Progress**:
   The system will display real-time progress updates:
   ```
   Processing: Frame 150/750 (20.0%) | Active Tracks: 8 | Total Sequential IDs: 12
   ```

### Advanced Configuration

You can modify tracking parameters in the `PlayerTracker` class:

```python
# Tracking sensitivity
self.confidence_threshold = 0.25  # Lower = more detections

# Track merging parameters
self.max_merge_distance = 100     # Pixel distance for merging
self.min_track_length = 15        # Minimum frames for valid track
self.max_lost_frames = 30         # Frames before track deletion
```

### Custom File Paths

Update the paths in the `main()` function:

```python
MODEL_PATH = r"path/to/your/model.pt"
VIDEO_PATH = r"path/to/your/video.mp4"
```

## 📊 Output Analysis

### Video Output
The system generates `output_tracking.mp4` with:
- **Bounding boxes**: Color-coded for each player
- **Sequential IDs**: Clean numbering (1, 2, 3...)
- **Confidence scores**: Detection confidence for each player
- **Motion trails**: Visual paths showing player movement
- **Real-time metrics**: On-screen tracking statistics

### Console Analytics

#### ID Management Summary
```
Sequential ID Management Summary:
- Original YOLO IDs encountered: 45
- Sequential IDs created: 28
- Final sequential tracks after merging: 22
- ID management efficiency: 78.6%
```

#### Quality Metrics
```
Tracking Quality Analysis:
- Total sequential tracks: 22
- Long-duration tracks (>3s): 18
- Very long-duration tracks (>6s): 12
- Excellent tracks (>10s): 8
- Track persistence rate: 81.8%
- Track fragmentation ratio: 1.2x
```

#### Performance Score
```
Overall Quality Score: 87/100
Rating: EXCELLENT - Outstanding sequential tracking performance
```

## 📁 Project Structure

```
Soccer-Player-Identification/
│
├── track_player.py          # Main tracking system
├── best.pt                  # Pre-trained YOLO model
├── 15sec_input_720p.mp4    # Sample input video
├── output_tracking.mp4     # Generated output video
├── requirements.txt         # Python dependencies
├── README.md               # This documentation
├── .gitignore             # Git ignore rules
├── .venv/                 # Virtual environment
└── .git/                  # Git repository
```

## 📈 Performance Metrics

### Quality Indicators

1. **ID Management Efficiency**: Ratio of final tracks to original YOLO IDs
2. **Track Persistence Rate**: Percentage of long-duration tracks
3. **Fragmentation Ratio**: How much track splitting occurs
4. **Overall Quality Score**: Composite score (0-100)

### Scoring System

- **85-100**: EXCELLENT - Outstanding tracking performance
- **70-84**: GOOD - Solid tracking with good results
- **50-69**: FAIR - Acceptable tracking, room for improvement
- **Below 50**: POOR - Significant improvement needed

### Expected Performance
For a typical soccer match video:
- **Track Persistence**: 75-85%
- **ID Efficiency**: 70-90%
- **Quality Score**: 80-95
- **Processing Speed**: 15-25 FPS (CPU), 30-60 FPS (GPU)

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Error
```
Error: Could not load model from best.pt
```
**Solution**: Ensure the model file exists and is a valid YOLO model trained for person/player detection.

#### 2. Video File Error
```
Error: Could not open video file
```
**Solution**: 
- Check file path is correct
- Ensure video format is supported (mp4, avi, mov)
- Verify file permissions

#### 3. Low Detection Performance
**Symptoms**: Few players detected, poor tracking
**Solutions**:
- Lower `confidence_threshold` (try 0.15-0.2)
- Check if model is trained for soccer players
- Verify video quality and resolution

#### 4. High Track Fragmentation
**Symptoms**: Many short tracks, low persistence rate
**Solutions**:
- Increase `max_merge_distance` (try 150-200)
- Increase `max_lost_frames` (try 45-60)
- Lower confidence threshold for better detection

#### 5. Memory Issues
**Symptoms**: System crashes or slow performance
**Solutions**:
- Reduce video resolution
- Process shorter video segments
- Close other applications
- Use GPU if available

### Performance Optimization

#### For Better Accuracy
```python
self.confidence_threshold = 0.2      # Lower threshold
self.max_merge_distance = 150        # More aggressive merging
self.max_lost_frames = 45            # Longer track retention
```

#### For Faster Processing
```python
self.confidence_threshold = 0.35     # Higher threshold
self.max_merge_distance = 80         # Less aggressive merging
```

## 🎓 Educational Value

This project demonstrates several important computer vision concepts:

1. **Object Detection**: Using YOLO for real-time player detection
2. **Object Tracking**: Maintaining identity across video frames
3. **Data Association**: Linking detections to create consistent tracks
4. **Post-processing**: Merging and refining tracking results
5. **Performance Evaluation**: Comprehensive metrics and analysis

## 🚀 Future Enhancements

Potential improvements for advanced users:

1. **Team Classification**: Distinguish between different teams
2. **Jersey Number Recognition**: OCR for jersey numbers
3. **Action Recognition**: Detect player actions (running, kicking, etc.)
4. **Heat Maps**: Generate player movement heat maps
5. **Performance Analytics**: Calculate player statistics

## 📝 Contributing

This project is designed for educational and internship purposes. Feel free to:
- Experiment with different parameters
- Add new features
- Improve tracking algorithms
- Enhance visualization

## 📄 License

This project is open source and available for educational use.

---

**Note**: This system is designed for educational purposes and sports analysis. Ensure you have appropriate permissions for any video content you process.
