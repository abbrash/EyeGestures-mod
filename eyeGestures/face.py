"""Module providing finding and extraction of face from image."""

import cv2
import numpy as np
import mediapipe as mp
import eyeGestures.eye as eye
from eyeGestures.head_pose import HeadPoseCompensator


class FaceFinder:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def find(self, image):

        assert (len(image.shape) > 2)

        try:
            face_mesh = self.mp_face_mesh.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if face_mesh.multi_face_landmarks is None:
                return None

            return face_mesh
        except Exception as e:
            print(f"Exception in FaceFinder: {e}")
            return None


class Face:

    def __init__(self):
        self.eyeLeft = eye.Eye(0)
        self.eyeRight = eye.Eye(1)
        self.landmarks = None
        self.head_pose_compensator = HeadPoseCompensator()
        self.head_pose_data = None
        self.reference_set = False

    def getBoundingBox(self):
        if self.landmarks is not None:
            margin = 0
            min_x = np.min(self.landmarks[:, 0]) - margin
            max_x = np.max(self.landmarks[:, 0]) + margin
            min_y = np.min(self.landmarks[:, 1]) - margin
            max_y = np.max(self.landmarks[:, 1]) + margin

            width = int((max_x - min_x))
            height = int((max_y - min_y))
            x = int(min_x)
            y = int(min_y)
            return (x, y, width, height)
        return (0, 0, 0, 0)

    def getLeftEye(self):
        return self.eyeLeft

    def getRightEye(self):
        return self.eyeRight

    def getLandmarks(self):
        return self.landmarks
    
    def getHeadPoseData(self):
        """Get current head pose compensation data."""
        return self.head_pose_data
    
    def setReferencePose(self):
        """Set the current pose as reference for compensation."""
        if self.landmarks is not None:
            success = self.head_pose_compensator.set_reference_pose(
                self.landmarks, self.image_w, self.image_h
            )
            self.reference_set = success
            return success
        return False
    
    def resetReferencePose(self):
        """Reset the reference pose."""
        self.head_pose_compensator.reset_reference()
        self.reference_set = False

    def _landmarks(self, face):

        __complex_landmark_points = face.multi_face_landmarks
        __complex_landmarks = __complex_landmark_points[0].landmark

        __face_landmarks = []
        for landmark in __complex_landmarks:
            __face_landmarks.append((
                landmark.x * self.image_w,
                landmark.y * self.image_h))

        return np.array(__face_landmarks)

    def process(self, image, face):
        # try:
        self.face = face
        self.image_h, self.image_w, _ = image.shape
        self.landmarks = self._landmarks(self.face)
        
        # Estimate head pose and get compensation data
        self.head_pose_data = self.head_pose_compensator.get_pose_compensation(
            self.landmarks, self.image_w, self.image_h
        )
        
        # Set reference pose if not already set
        if not self.reference_set and self.head_pose_data['success']:
            self.setReferencePose()

        x, y, _, _ = self.getBoundingBox()
        offset = np.array((x, y))
        
        # Apply head pose compensation to offset if available
        if self.head_pose_data['success']:
            # Adjust offset based on head pose
            tilt_compensation = self.head_pose_data['tilt_angle'] * 0.1
            offset[0] += tilt_compensation
            offset[1] += self.head_pose_data['compensation_y'] * 0.05

        self.eyeLeft.update(image, self.landmarks, offset)
        self.eyeRight.update(image, self.landmarks, offset)
        # except Exception as e:
        #     print(f"Caught exception: {e}")
        #     return None
