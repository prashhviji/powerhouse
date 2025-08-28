import cv2
import numpy as np
import mediapipe as mp
import json
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ExerciseType(Enum):
    """Define different physiotherapy exercises."""
    SHOULDER_RAISE = "shoulder_raise"
    ARM_STRETCH = "arm_stretch"
    SQUAT = "squat"
    STANDING_BALANCE = "standing_balance"
    NECK_ROTATION = "neck_rotation"
    LEFT_ARM_RAISE = "left_arm_raise"

@dataclass
class PostureRule:
    """Define a rule for correct posture."""
    joint1: str
    joint2: str
    joint3: str
    angle_range: Tuple[float, float]  # (min, max) acceptable angles
    description: str
    weight: float = 1.0  # Importance weight

@dataclass
class PostureScore:
    """Store posture evaluation results."""
    overall_score: float
    individual_scores: Dict[str, float]
    feedback_messages: List[str]
    is_correct: bool

class PhysiotherapyPostureChecker:
    def __init__(self):
        """Initialize the posture checker system."""
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Define landmark indices for easier access
        self.landmark_names = {
            'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13,
            'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
            'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29,
            'right_heel': 30, 'left_foot_index': 31, 'right_foot_index': 32
        }
        
        # Initialize exercise rules
        self.exercise_rules = self._initialize_exercise_rules()
        self.current_exercise = ExerciseType.LEFT_ARM_RAISE  # Start with left arm raise
        
        # Tracking variables
        self.pose_history = []
        self.max_history = 30  # frames
        self.correct_pose_threshold = 0.8
        
    def _initialize_exercise_rules(self) -> Dict[ExerciseType, List[PostureRule]]:
        """Initialize posture rules for different exercises."""
        rules = {}
        
        # Shoulder Raise Exercise
        rules[ExerciseType.SHOULDER_RAISE] = [
            PostureRule(
                joint1="left_shoulder", joint2="left_elbow", joint3="left_wrist",
                angle_range=(160, 180), description="Left arm should be straight up",
                weight=1.5
            ),
            PostureRule(
                joint1="right_shoulder", joint2="right_elbow", joint3="right_wrist",
                angle_range=(160, 180), description="Right arm should be straight up",
                weight=1.5
            ),
            PostureRule(
                joint1="left_shoulder", joint2="nose", joint3="right_shoulder",
                angle_range=(170, 190), description="Keep shoulders level",
                weight=1.0
            )
        ]
        
        # Squat Exercise
        rules[ExerciseType.SQUAT] = [
            PostureRule(
                joint1="left_hip", joint2="left_knee", joint3="left_ankle",
                angle_range=(80, 110), description="Left knee bent at proper angle",
                weight=2.0
            ),
            PostureRule(
                joint1="right_hip", joint2="right_knee", joint3="right_ankle",
                angle_range=(80, 110), description="Right knee bent at proper angle",
                weight=2.0
            ),
            PostureRule(
                joint1="left_shoulder", joint2="left_hip", joint3="left_knee",
                angle_range=(160, 190), description="Keep back straight",
                weight=1.5
            )
        ]
        
        # Arm Stretch Exercise
        rules[ExerciseType.ARM_STRETCH] = [
            PostureRule(
                joint1="left_shoulder", joint2="left_elbow", joint3="left_wrist",
                angle_range=(170, 190), description="Left arm should be extended",
                weight=1.0
            ),
            PostureRule(
                joint1="right_shoulder", joint2="right_elbow", joint3="right_wrist",
                angle_range=(170, 190), description="Right arm should be extended",
                weight=1.0
            ),
            PostureRule(
                joint1="left_wrist", joint2="left_shoulder", joint3="right_shoulder",
                angle_range=(85, 95), description="Arms should be perpendicular to body",
                weight=1.2
            )
        ]
        
        # Left Arm Raise Exercise
        rules[ExerciseType.LEFT_ARM_RAISE] = [
            PostureRule(
                joint1="left_shoulder", joint2="left_elbow", joint3="left_wrist",
                angle_range=(160, 180), description="Left arm should be fully extended",
                weight=2.0
            ),
            PostureRule(
                joint1="left_hip", joint2="left_shoulder", joint3="left_elbow",
                angle_range=(80, 100), description="Left arm should be raised to shoulder height or above",
                weight=2.5
            ),
            PostureRule(
                joint1="left_shoulder", joint2="nose", joint3="right_shoulder",
                angle_range=(165, 195), description="Keep shoulders level - don't lean",
                weight=1.5
            ),
            PostureRule(
                joint1="right_shoulder", joint2="right_elbow", joint3="right_wrist",
                angle_range=(160, 200), description="Right arm should remain relaxed at side",
                weight=1.0
            ),
            PostureRule(
                joint1="right_hip", joint2="right_shoulder", joint3="right_elbow",
                angle_range=(160, 200), description="Right arm should stay down",
                weight=1.2
            )
        ]
        
        return rules
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                       p3: Tuple[float, float]) -> float:
        """Calculate angle between three points."""
        # Convert to numpy arrays
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        angle = np.arccos(cos_angle)
        
        return math.degrees(angle)
    
    def extract_landmarks(self, pose_results) -> Optional[Dict[str, Tuple[float, float]]]:
        """Extract landmark coordinates from pose results."""
        if not pose_results.pose_landmarks:
            return None
        
        landmarks = {}
        for name, idx in self.landmark_names.items():
            landmark = pose_results.pose_landmarks.landmark[idx]
            if landmark.visibility > 0.5:  # Only use visible landmarks
                landmarks[name] = (landmark.x, landmark.y)
        
        return landmarks
    
    def evaluate_posture_rule(self, landmarks: Dict[str, Tuple[float, float]], 
                             rule: PostureRule) -> Tuple[float, str]:
        """Evaluate a single posture rule."""
        # Check if all required landmarks are available
        if not all(joint in landmarks for joint in [rule.joint1, rule.joint2, rule.joint3]):
            return 0.0, f"Cannot evaluate: {rule.description} (landmarks not visible)"
        
        # Calculate angle
        p1 = landmarks[rule.joint1]
        p2 = landmarks[rule.joint2]
        p3 = landmarks[rule.joint3]
        
        current_angle = self.calculate_angle(p1, p2, p3)
        
        # Check if angle is within acceptable range
        min_angle, max_angle = rule.angle_range
        
        if min_angle <= current_angle <= max_angle:
            score = 1.0
            feedback = f"✓ {rule.description} (angle: {current_angle:.1f}°)"
        else:
            # Calculate how far off the angle is
            if current_angle < min_angle:
                distance = min_angle - current_angle
                feedback = f"✗ {rule.description} - increase angle by {distance:.1f}° (current: {current_angle:.1f}°)"
            else:
                distance = current_angle - max_angle
                feedback = f"✗ {rule.description} - decrease angle by {distance:.1f}° (current: {current_angle:.1f}°)"
            
            # Score based on how close to acceptable range
            max_distance = max(abs(current_angle - min_angle), abs(current_angle - max_angle))
            score = max(0.0, 1.0 - (max_distance / 90.0))  # Normalize by 90 degrees
        
        return score, feedback
    
    def evaluate_posture(self, pose_results) -> PostureScore:
        """Evaluate current posture against exercise rules."""
        landmarks = self.extract_landmarks(pose_results)
        
        if not landmarks:
            return PostureScore(
                overall_score=0.0,
                individual_scores={},
                feedback_messages=["No pose detected"],
                is_correct=False
            )
        
        # Get rules for current exercise
        rules = self.exercise_rules.get(self.current_exercise, [])
        
        if not rules:
            return PostureScore(
                overall_score=0.0,
                individual_scores={},
                feedback_messages=["No rules defined for this exercise"],
                is_correct=False
            )
        
        # Evaluate each rule
        individual_scores = {}
        feedback_messages = []
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for i, rule in enumerate(rules):
            score, feedback = self.evaluate_posture_rule(landmarks, rule)
            individual_scores[f"rule_{i}"] = score
            feedback_messages.append(feedback)
            
            total_weighted_score += score * rule.weight
            total_weight += rule.weight
        
        # Calculate overall score
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        is_correct = overall_score >= self.correct_pose_threshold
        
        return PostureScore(
            overall_score=overall_score,
            individual_scores=individual_scores,
            feedback_messages=feedback_messages,
            is_correct=is_correct
        )
    
    def draw_feedback(self, image: np.ndarray, score: PostureScore) -> np.ndarray:
        """Draw feedback on the image."""
        h, w, _ = image.shape
        
        # Draw overall score
        score_color = (0, 255, 0) if score.is_correct else (0, 0, 255)
        score_text = f"Score: {score.overall_score:.2f}"
        status_text = "CORRECT POSTURE" if score.is_correct else "INCORRECT POSTURE"
        
        cv2.putText(image, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, score_color, 2)
        cv2.putText(image, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, score_color, 2)
        
        # Draw exercise name
        exercise_name = self.current_exercise.value.replace('_', ' ').title()
        cv2.putText(image, f"Exercise: {exercise_name}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw feedback messages
        y_offset = 120
        for msg in score.feedback_messages[:5]:  # Show only first 5 messages
            color = (0, 255, 0) if msg.startswith("✓") else (0, 100, 255)
            cv2.putText(image, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
            y_offset += 25
        
        # Draw score bar
        bar_width = 200
        bar_height = 20
        bar_x = w - bar_width - 20
        bar_y = 30
        
        # Background bar
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Score bar
        score_width = int(bar_width * score.overall_score)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + score_width, bar_y + bar_height), 
                     score_color, -1)
        
        # Bar text
        cv2.putText(image, "Posture Score", (bar_x, bar_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, PostureScore]:
        """Process a single frame for posture analysis."""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Evaluate posture
        score = self.evaluate_posture(results)
        
        # Draw feedback
        frame = self.draw_feedback(frame, score)
        
        # Update pose history
        self.pose_history.append(score.overall_score)
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
        
        return frame, score
    
    def change_exercise(self, exercise_type: ExerciseType):
        """Change the current exercise being evaluated."""
        self.current_exercise = exercise_type
        self.pose_history.clear()
        print(f"Changed exercise to: {exercise_type.value.replace('_', ' ').title()}")
    
    def run_live_assessment(self, camera_index: int = 0):
        """Run live posture assessment."""
        cap = cv2.VideoCapture( "http://172.16.40.239:8080/video" )
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Physiotherapy Posture Checker")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press '1' for Shoulder Raise")
        print("- Press '2' for Squat")
        print("- Press '3' for Arm Stretch")
        print("- Press '4' for Left Arm Raise")
        print("- Press 's' to save current analysis")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                annotated_frame, score = self.process_frame(frame)
                
                # Display
                cv2.imshow('Physiotherapy Posture Checker', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    self.change_exercise(ExerciseType.SHOULDER_RAISE)
                elif key == ord('2'):
                    self.change_exercise(ExerciseType.SQUAT)
                elif key == ord('3'):
                    self.change_exercise(ExerciseType.ARM_STRETCH)
                elif key == ord('4'):
                    self.change_exercise(ExerciseType.LEFT_ARM_RAISE)
                elif key == ord('s'):
                    # Save analysis
                    cv2.imwrite(f'posture_analysis_{self.current_exercise.value}.jpg', 
                               annotated_frame)
                    print(f"Analysis saved for {self.current_exercise.value}")
        
        except KeyboardInterrupt:
            print("\nAssessment interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()

def main():
    """Main function."""
    checker = PhysiotherapyPostureChecker()
    checker.run_live_assessment()

if __name__ == "__main__":
    try:
        import mediapipe
        import cv2
        import numpy
    except ImportError as e:
        print(f"Required package not found: {e}")
        print("Please install: pip install mediapipe opencv-python numpy")
        exit(1)
    
    main()