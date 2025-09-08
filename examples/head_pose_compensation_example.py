import os
import sys
import cv2
import pygame
import numpy as np

pygame.init()
pygame.font.init()

def get_gaze_zone(gaze_point, screen_width, screen_height):
    """
    Determine which of the 9 zones the gaze point is located in.
    Zones are arranged in a 3x3 grid:
    1 2 3
    4 5 6
    7 8 9
    """
    x, y = gaze_point
    
    # Calculate zone boundaries
    zone_width = screen_width / 3
    zone_height = screen_height / 3
    
    # Determine column (1, 2, or 3)
    if x < zone_width:
        col = 1
    elif x < 2 * zone_width:
        col = 2
    else:
        col = 3
    
    # Determine row (1, 2, or 3)
    if y < zone_height:
        row = 1
    elif y < 2 * zone_height:
        row = 2
    else:
        row = 3
    
    # Calculate zone number (1-9)
    zone = (row - 1) * 3 + col
    
    return zone, (col - 1) * zone_width, (row - 1) * zone_height, zone_width, zone_height

# Get the display dimensions
screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

# Set up the screen
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("EyeGestures v3 with Head Pose Compensation")
font_size = 48
bold_font = pygame.font.Font(None, font_size)
bold_font.set_bold(True)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/..')

from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3

# Initialize gesture engine with head pose compensation
gestures = EyeGestures_v3()
cap = VideoCapture(0)

# Enable head pose compensation (enabled by default)
gestures.enableHeadPoseCompensation(True)

# Create calibration points
x = np.arange(0, 1.1, 0.2)
y = np.arange(0, 1.1, 0.2)
xx, yy = np.meshgrid(x, y)
calibration_map = np.column_stack([xx.ravel(), yy.ravel()])
n_points = min(len(calibration_map), 25)
np.random.shuffle(calibration_map)
gestures.uploadCalibrationMap(calibration_map, context="head_pose_demo")
gestures.setFixation(1.0)

# Colors
RED = (255, 0, 100)
BLUE = (100, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

clock = pygame.time.Clock()

# Main game loop
running = True
iterator = 0
prev_x = 0
prev_y = 0
reference_set = False

print("Head Pose Compensation Demo")
print("==========================")
print("Instructions:")
print("1. Look at the calibration points when they appear")
print("2. After calibration, try moving your head left/right and tilting")
print("3. Notice how gaze tracking remains accurate despite head movement")
print("4. Press 'R' to reset head pose reference")
print("5. Press 'T' to toggle head pose compensation")
print("6. Press Ctrl+Q to quit")

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL:
                running = False
            elif event.key == pygame.K_r:
                # Reset head pose reference
                gestures.resetHeadPoseReference()
                reference_set = False
                print("Head pose reference reset!")
            elif event.key == pygame.K_t:
                # Toggle head pose compensation
                current_state = gestures.enable_head_pose_compensation
                gestures.enableHeadPoseCompensation(not current_state)
                print(f"Head pose compensation: {'ON' if not current_state else 'OFF'}")

    # Generate new random position for the cursor
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.flip(frame, axis=1)
    
    calibrate = (iterator <= n_points)  # calibrate points
    event, calibration = gestures.step(frame, calibrate, screen_width, screen_height, context="head_pose_demo")

    if event is None:
        continue

    screen.fill(BLACK)
    
    # Display camera feed
    frame_surface = pygame.surfarray.make_surface(frame)
    frame_surface = pygame.transform.scale(frame_surface, (400, 300))
    screen.blit(frame_surface, (0, 0))

    # Display head pose information
    head_pose_data = gestures.face.getHeadPoseData()
    if head_pose_data and head_pose_data['success']:
        info_font = pygame.font.SysFont('Arial', 20)
        
        # Head pose angles
        angles = head_pose_data.get('raw_angles', [0, 0, 0])
        pitch, yaw, roll = angles[0], angles[1], angles[2]
        
        # Compensation values
        comp_x = head_pose_data.get('compensation_x', 0)
        comp_y = head_pose_data.get('compensation_y', 0)
        tilt = head_pose_data.get('tilt_angle', 0)
        
        # Display head pose information
        y_offset = 320
        screen.blit(info_font.render(f"Head Pose Angles:", True, WHITE), (10, y_offset))
        screen.blit(info_font.render(f"Pitch: {pitch:.1f}°", True, WHITE), (10, y_offset + 25))
        screen.blit(info_font.render(f"Yaw: {yaw:.1f}°", True, WHITE), (10, y_offset + 50))
        screen.blit(info_font.render(f"Roll: {roll:.1f}°", True, WHITE), (10, y_offset + 75))
        
        screen.blit(info_font.render(f"Compensation:", True, WHITE), (10, y_offset + 110))
        screen.blit(info_font.render(f"X: {comp_x:.2f}", True, WHITE), (10, y_offset + 135))
        screen.blit(info_font.render(f"Y: {comp_y:.2f}", True, WHITE), (10, y_offset + 160))
        screen.blit(info_font.render(f"Tilt: {tilt:.1f}°", True, WHITE), (10, y_offset + 185))
        
        # Status indicators
        status_color = GREEN if gestures.enable_head_pose_compensation else RED
        screen.blit(info_font.render(f"Head Pose Comp: {'ON' if gestures.enable_head_pose_compensation else 'OFF'}", True, status_color), (10, y_offset + 220))
        
        reference_color = GREEN if reference_set else YELLOW
        screen.blit(info_font.render(f"Reference: {'SET' if reference_set else 'NOT SET'}", True, reference_color), (10, y_offset + 245))

    if event is not None or calibration is not None:
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render(f'Fixation: {event.fixation:.2f}', False, WHITE)
        screen.blit(text_surface, (420, 0))
        
        if calibrate:
            if calibration.point[0] != prev_x or calibration.point[1] != prev_y:
                iterator += 1
                prev_x = calibration.point[0]
                prev_y = calibration.point[1]
            
            pygame.draw.circle(screen, BLUE, calibration.point, calibration.acceptance_radius)
            text_surface = bold_font.render(f"{iterator}/{n_points}", True, WHITE)
            text_square = text_surface.get_rect(center=calibration.point)
            screen.blit(text_surface, text_square)
        else:
            # Set reference pose after calibration
            if not reference_set and iterator > n_points:
                gestures.face.setReferencePose()
                reference_set = True
                print("Head pose reference set!")
        
        # Draw gaze point with different colors based on algorithm
        algorithm = gestures.whichAlgorithm(context="head_pose_demo")
        if algorithm == "Ridge":
            pygame.draw.circle(screen, RED, event.point, 50)
        elif algorithm == "LassoCV":
            pygame.draw.circle(screen, BLUE, event.point, 50)
        
        if event.saccades:
            pygame.draw.circle(screen, GREEN, event.point, 30)
        
        # Zone detection and visualization
        if not calibrate and reference_set:
            zone, zone_x, zone_y, zone_width, zone_height = get_gaze_zone(event.point, screen_width, screen_height)
            
            # Create a semi-transparent overlay for the active zone
            zone_surface = pygame.Surface((zone_width, zone_height))
            zone_surface.set_alpha(100)
            zone_surface.fill(YELLOW)
            screen.blit(zone_surface, (zone_x, zone_y))
            
            # Draw zone boundaries
            for i in range(4):
                pygame.draw.line(screen, WHITE, (i * zone_width, 0), (i * zone_width, screen_height), 2)
                pygame.draw.line(screen, WHITE, (0, i * zone_height), (screen_width, i * zone_height), 2)
            
            # Display zone number
            zone_font = pygame.font.SysFont('Comic Sans MS', 60)
            zone_text = zone_font.render(f'Zone {zone}', True, WHITE)
            text_rect = zone_text.get_rect(center=(zone_x + zone_width/2, zone_y + zone_height/2))
            screen.blit(zone_text, text_rect)

        # Display algorithm information
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render(f'Algorithm: {algorithm}', False, WHITE)
        screen.blit(text_surface, (420, 40))

    # Display instructions
    instruction_font = pygame.font.SysFont('Arial', 16)
    instructions = [
        "R: Reset head pose reference",
        "T: Toggle head pose compensation", 
        "Ctrl+Q: Quit"
    ]
    for i, instruction in enumerate(instructions):
        screen.blit(instruction_font.render(instruction, True, WHITE), (420, 300 + i * 20))

    pygame.display.flip()
    clock.tick(60)

# Cleanup
pygame.quit()
cap.close()
