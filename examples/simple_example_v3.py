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
pygame.display.set_caption("EyeGestures v3 example")
font_size = 48
bold_font = pygame.font.Font(None, font_size)
bold_font.set_bold(True)  # Set the font to bold

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/..')

from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3

gestures = EyeGestures_v3()
cap = VideoCapture(0)

x = np.arange(0, 1.1, 0.2)
y = np.arange(0, 1.1, 0.2)

xx, yy = np.meshgrid(x, y)

targets = [
    (0.6,0.3,0.1,0.05,"target_1"),
    (0.8,0.6,0.15,0.1,"target_2"),
    (0.1,0.9,0.1,0.1,"target_3")
]

calibration_map = np.column_stack([xx.ravel(), yy.ravel()])
n_points = min(len(calibration_map),25)
np.random.shuffle(calibration_map)
gestures.uploadCalibrationMap(calibration_map,context="my_context")
gestures.setFixation(1.0)
# Initialize Pygame
# Set up colors
RED = (255, 0, 100)
BLUE = (100, 0, 255)
GREEN = (0, 255, 0)
BLANK = (0,0,0)
WHITE = (255, 255, 255)

clock = pygame.time.Clock()

# Main game loop
running = True
iterator = 0
prev_x = 0
prev_y = 0
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL:
                running = False


    # Generate new random position for the cursor
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # frame = np.rot90(frame)
    frame = np.flip(frame, axis=1)
    calibrate = (iterator <= n_points) # calibrate 25 points
    event, calibration = gestures.step(frame, calibrate, screen_width, screen_height, context="my_context")

    if event is None:
        continue


    screen.fill((0, 0, 0))
    frame = pygame.surfarray.make_surface(frame)
    frame = pygame.transform.scale(frame, (400, 400))

    if event is not None or calibration is not None:
        # Display frame on Pygame screen
        screen.blit(
            pygame.surfarray.make_surface(
                np.rot90(event.sub_frame)
            ),
            (0, 0)
        )
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render(f'{event.fixation}', False, (0, 0, 0))
        screen.blit(text_surface, (0,0))
        if calibrate:
            if calibration.point[0] != prev_x or calibration.point[1] != prev_y:
                iterator += 1
                prev_x = calibration.point[0]
                prev_y = calibration.point[1]
            # pygame.draw.circle(screen, GREEN, fit_point, calibration_radius)
            pygame.draw.circle(screen, BLUE, calibration.point, calibration.acceptance_radius)
            text_surface = bold_font.render(f"{iterator}/{n_points}", True, WHITE)
            text_square = text_surface.get_rect(center=calibration.point)
            screen.blit(text_surface, text_square)
        else:
            pass
        # Draw gaze point
        if gestures.whichAlgorithm(context="my_context") == "Ridge":
            pygame.draw.circle(screen, RED, event.point, 50)
        if gestures.whichAlgorithm(context="my_context") == "LassoCV":
            pygame.draw.circle(screen, BLUE, event.point, 50)
        if event.saccades:
            pygame.draw.circle(screen, GREEN, event.point, 50)
        
        # Zone detection and visualization
        if not calibrate:  # Only show zones when not calibrating
            zone, zone_x, zone_y, zone_width, zone_height = get_gaze_zone(event.point, screen_width, screen_height)
            
            # Create a semi-transparent overlay for the active zone
            zone_surface = pygame.Surface((zone_width, zone_height))
            zone_surface.set_alpha(100)  # Semi-transparent
            zone_surface.fill((255, 255, 0))  # Yellow illumination
            screen.blit(zone_surface, (zone_x, zone_y))
            
            # Draw zone boundaries
            for i in range(4):  # 3x3 grid needs 4 lines each direction
                # Vertical lines
                pygame.draw.line(screen, WHITE, (i * zone_width, 0), (i * zone_width, screen_height), 2)
                # Horizontal lines
                pygame.draw.line(screen, WHITE, (0, i * zone_height), (screen_width, i * zone_height), 2)
            
            # Display zone number
            zone_font = pygame.font.SysFont('Comic Sans MS', 60)
            zone_text = zone_font.render(f'Zone {zone}', True, WHITE)
            text_rect = zone_text.get_rect(center=(zone_x + zone_width/2, zone_y + zone_height/2))
            screen.blit(zone_text, text_rect)

        for target in targets:
            pygame.draw.rect(
                screen,
                RED,
                pygame.Rect(
                int(target[0] * screen_width),
                int(target[1] * screen_height),
                int(target[2] * screen_width),
                int(target[3] * screen_height),
            ))
            text_surface = my_font.render(f'{target[4]}', False, (0, 0, 0))
            screen.blit(text_surface, (int(target[0] * screen_width),int(target[1] * screen_height)))


        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render(f'{gestures.whichAlgorithm(context="my_context")}', False, (0, 0, 0))
        screen.blit(text_surface, event.point)

    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
