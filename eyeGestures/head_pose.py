"""Module providing head pose estimation and compensation for gaze tracking."""

import cv2
import numpy as np
import mediapipe as mp


class HeadPoseEstimator:
    """Class for estimating 3D head pose from facial landmarks."""
    
    def __init__(self):
        # 3D model points for head pose estimation (in mm)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # MediaPipe landmark indices for the model points
        self.landmark_indices = [1, 152, 33, 263, 61, 291]  # Nose, chin, left eye, right eye, left mouth, right mouth
        
        # Camera matrix (will be estimated from image dimensions)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
    def _get_camera_matrix(self, image_width, image_height):
        """Estimate camera matrix from image dimensions."""
        if self.camera_matrix is None:
            # Estimate focal length (typical for webcams)
            focal_length = max(image_width, image_height)
            center_x = image_width / 2.0
            center_y = image_height / 2.0
            
            self.camera_matrix = np.array([
                [focal_length, 0, center_x],
                [0, focal_length, center_y],
                [0, 0, 1]
            ], dtype=np.float64)
        
        return self.camera_matrix
    
    def estimate_pose(self, landmarks, image_width, image_height):
        """Estimate 3D head pose from 2D landmarks."""
        try:
            # Extract 2D image points
            image_points = np.array([
                landmarks[self.landmark_indices[0]],  # Nose tip
                landmarks[self.landmark_indices[1]],  # Chin
                landmarks[self.landmark_indices[2]],  # Left eye
                landmarks[self.landmark_indices[3]],  # Right eye
                landmarks[self.landmark_indices[4]],  # Left mouth
                landmarks[self.landmark_indices[5]]   # Right mouth
            ], dtype=np.float64)
            
            # Get camera matrix
            camera_matrix = self._get_camera_matrix(image_width, image_height)
            
            # Solve PnP to get rotation and translation vectors
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Extract Euler angles (pitch, yaw, roll)
                euler_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
                
                return {
                    'rotation_vector': rotation_vector,
                    'translation_vector': translation_vector,
                    'rotation_matrix': rotation_matrix,
                    'euler_angles': euler_angles,  # [pitch, yaw, roll] in degrees
                    'success': True
                }
            else:
                return {'success': False}
                
        except Exception as e:
            print(f"Head pose estimation error: {e}")
            return {'success': False}
    
    def _rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles (pitch, yaw, roll)."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([np.degrees(x), np.degrees(y), np.degrees(z)])


class HeadPoseCompensator:
    """Class for compensating gaze estimation based on head pose."""
    
    def __init__(self):
        self.pose_estimator = HeadPoseEstimator()
        self.reference_pose = None
        self.pose_history = []
        self.max_history = 10
        
    def set_reference_pose(self, landmarks, image_width, image_height):
        """Set the reference head pose for compensation."""
        pose_data = self.pose_estimator.estimate_pose(landmarks, image_width, image_height)
        if pose_data['success']:
            self.reference_pose = pose_data
            return True
        return False
    
    def get_pose_compensation(self, landmarks, image_width, image_height):
        """Get compensation factors based on current head pose."""
        current_pose = self.pose_estimator.estimate_pose(landmarks, image_width, image_height)
        
        if not current_pose['success'] or self.reference_pose is None:
            return {
                'compensation_x': 0.0,
                'compensation_y': 0.0,
                'compensation_scale': 1.0,
                'tilt_angle': 0.0,
                'success': False
            }
        
        # Add to pose history for smoothing
        self.pose_history.append(current_pose['euler_angles'])
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
        
        # Smooth the pose angles
        smoothed_angles = np.mean(self.pose_history, axis=0)
        
        # Calculate differences from reference pose
        ref_angles = self.reference_pose['euler_angles']
        angle_diffs = smoothed_angles - ref_angles
        
        # Extract individual angle differences
        pitch_diff = angle_diffs[0]  # Up/down head movement
        yaw_diff = angle_diffs[1]    # Left/right head movement
        roll_diff = angle_diffs[2]   # Head tilt
        
        # Calculate compensation factors
        # Yaw (left/right) affects X-axis gaze
        compensation_x = yaw_diff * 0.5  # Scale factor for sensitivity
        
        # Pitch (up/down) affects Y-axis gaze
        compensation_y = pitch_diff * 0.3  # Scale factor for sensitivity
        
        # Roll (tilt) affects overall scale and requires rotation compensation
        tilt_angle = roll_diff
        compensation_scale = 1.0 + abs(tilt_angle) * 0.01  # Slight scale adjustment
        
        return {
            'compensation_x': compensation_x,
            'compensation_y': compensation_y,
            'compensation_scale': compensation_scale,
            'tilt_angle': tilt_angle,
            'success': True,
            'raw_angles': smoothed_angles,
            'angle_diffs': angle_diffs
        }
    
    def apply_compensation(self, gaze_point, compensation_data):
        """Apply head pose compensation to gaze point."""
        if not compensation_data['success']:
            return gaze_point
        
        x, y = gaze_point
        
        # Apply tilt rotation (simplified 2D rotation)
        tilt_angle_rad = np.radians(compensation_data['tilt_angle'])
        cos_tilt = np.cos(tilt_angle_rad)
        sin_tilt = np.sin(tilt_angle_rad)
        
        # Rotate gaze point around center
        center_x, center_y = 0.5, 0.5  # Assume normalized coordinates
        x_centered = x - center_x
        y_centered = y - center_y
        
        # Apply rotation
        x_rotated = x_centered * cos_tilt - y_centered * sin_tilt
        y_rotated = x_centered * sin_tilt + y_centered * cos_tilt
        
        # Apply compensation
        x_compensated = x_rotated + center_x - compensation_data['compensation_x'] * 0.1
        y_compensated = y_rotated + center_y - compensation_data['compensation_y'] * 0.1
        
        # Apply scale compensation
        x_final = (x_compensated - center_x) * compensation_data['compensation_scale'] + center_x
        y_final = (y_compensated - center_y) * compensation_data['compensation_scale'] + center_y
        
        return np.array([x_final, y_final])
    
    def reset_reference(self):
        """Reset the reference pose."""
        self.reference_pose = None
        self.pose_history = []
