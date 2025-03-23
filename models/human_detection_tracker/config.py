# Config for Swimming Pool Monitoring System

# Model Configuration
MODEL = {
    'path': 'yolo11n.pt',
    'detection_threshold': 0.5,
    'class_filter': [0],
}

# Tracking Configuration
TRACKING = {
    'max_cosine_distance': 0.4,
    'nn_budget': 100,
    'max_iou_distance': 0.7,
    'max_age': 30,
    'missing_threshold': 30,
}

# Video Processing
VIDEO = {
    'input': 0,
    'output': 'pool_monitoring.mp4',
    'display': True,
    'display_rois': True,
}

# Alert System
ALERTS = {
    'enabled': True,  # Enable alerts
    'missing_alert': True,  # Alert when a person is missing for too long
    'missing_threshold_seconds': 10,  # Alert if person missing for this many seconds
    'save_missing_rois': True,  # Save ROIs of missing persons
    'save_path': 'missing_persons/',  # Path to save missing persons ROIs
}

# Advanced Options
ADVANCED = {
    'use_gpu': True,  # Use GPU for inference if available
    'frame_skip': 0,  # Process every Nth frame (0 = process all frames)
    'roi_scale': 1.0,  # Scale factor for ROIs (> 1.0 means larger ROIs)
    'track_history': 10,  # Number of previous positions to store for each track
}
