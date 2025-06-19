# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:35:13 2025

@author: ammar
"""

import numpy as np
import cv2

CENTER_X = 512
CENTER_Y = 384
SQUARE_SIZE = 50 # will be multiplied by 2
MARKER_SIZE = 10
CIRCLE_DIAMETER = 100
LINE_THICKNESS = 2

def draw_crosshair(image):
    color = (100, 100, 100)  # Line color in BGR, e.g., blue
    thickness = LINE_THICKNESS  # Line thickness in pixels

    # draw the x-y coordinate
    y1_axis_start = (CENTER_X, 0)  # Starting coordinate, e.g., (x1, y1)
    y1_axis_end = (CENTER_X, CENTER_Y-SQUARE_SIZE)  # Ending coordinate, e.g., (x2, y2)
    
    y2_axis_start = (CENTER_X, CENTER_Y+SQUARE_SIZE)  # Starting coordinate, e.g., (x1, y1)
    y2_axis_end = (CENTER_X, CENTER_Y*2)  # Ending coordinate, e.g., (x2, y2)
    
    x1_axis_start = (0, CENTER_Y)  # Starting coordinate, e.g., (x1, y1)
    x1_axis_end = (CENTER_X-SQUARE_SIZE, CENTER_Y)  # Ending coordinate, e.g., (x2, y2)
    
    x2_axis_start = (CENTER_X+SQUARE_SIZE, CENTER_Y)  # Starting coordinate, e.g., (x1, y1)
    x2_axis_end = ( CENTER_X*2, CENTER_Y)  # Ending coordinate, e.g., (x2, y2)
    
    image = cv2.line(image, y1_axis_start, y1_axis_end, color, thickness)
    image = cv2.line(image, y2_axis_start, y2_axis_end, color, thickness)
    
    image = cv2.line(image, x1_axis_start, x1_axis_end, color, thickness)
    image = cv2.line(image, x2_axis_start, x2_axis_end, color, thickness)
    
    # draw the rectangle
    rect11_start = (CENTER_X-SQUARE_SIZE, CENTER_Y-SQUARE_SIZE)  # Starting coordinate, e.g., (x1, y1)
    rect11_end = (CENTER_X-(SQUARE_SIZE-2*MARKER_SIZE), CENTER_Y-SQUARE_SIZE)  # Ending coordinate, e.g., (x2, y2)
    rect12_start = (CENTER_X-SQUARE_SIZE, CENTER_Y-SQUARE_SIZE)  # Starting coordinate, e.g., (x1, y1)
    rect12_end = (CENTER_X-SQUARE_SIZE, CENTER_Y-(SQUARE_SIZE-2*MARKER_SIZE))  # Ending coordinate, e.g., (x2, y2)
    
    rect21_start = (CENTER_X+(SQUARE_SIZE-2*MARKER_SIZE), CENTER_Y-SQUARE_SIZE)  # Starting coordinate, e.g., (x1, y1)
    rect21_end = (CENTER_X+SQUARE_SIZE, CENTER_Y-SQUARE_SIZE)  # Ending coordinate, e.g., (x2, y2)
    rect22_start = (CENTER_X+SQUARE_SIZE, CENTER_Y-SQUARE_SIZE)  # Starting coordinate, e.g., (x1, y1)
    rect22_end = (CENTER_X+SQUARE_SIZE, CENTER_Y-(SQUARE_SIZE-2*MARKER_SIZE))  # Ending coordinate, e.g., (x2, y2)
    
    rect31_start = (CENTER_X-SQUARE_SIZE, CENTER_Y+SQUARE_SIZE)  # Starting coordinate, e.g., (x1, y1)
    rect31_end = (CENTER_X-(SQUARE_SIZE-2*MARKER_SIZE), CENTER_Y+SQUARE_SIZE)  # Ending coordinate, e.g., (x2, y2)
    rect32_start = (CENTER_X-SQUARE_SIZE, CENTER_Y+(SQUARE_SIZE-2*MARKER_SIZE))  # Starting coordinate, e.g., (x1, y1)
    rect32_end = (CENTER_X-SQUARE_SIZE, CENTER_Y+SQUARE_SIZE)  # Ending coordinate, e.g., (x2, y2)
    
    rect41_start = (CENTER_X+(SQUARE_SIZE-2*MARKER_SIZE), CENTER_Y+SQUARE_SIZE)  # Starting coordinate, e.g., (x1, y1)
    rect41_end = (CENTER_X+SQUARE_SIZE, CENTER_Y+SQUARE_SIZE)  # Ending coordinate, e.g., (x2, y2)
    rect42_start = (CENTER_X+SQUARE_SIZE, CENTER_Y+(SQUARE_SIZE-2*MARKER_SIZE))  # Starting coordinate, e.g., (x1, y1)
    rect42_end = (CENTER_X+SQUARE_SIZE, CENTER_Y+SQUARE_SIZE)  # Ending coordinate, e.g., (x2, y2)
    
    # cross on center
    cross1_start = (CENTER_X, CENTER_Y-MARKER_SIZE)
    cross1_end = (CENTER_X, CENTER_Y+MARKER_SIZE)
    
    cross2_start = (CENTER_X-MARKER_SIZE, CENTER_Y)
    cross2_end = (CENTER_X+MARKER_SIZE, CENTER_Y)
    
    image = cv2.line(image, rect11_start, rect11_end, color, thickness)
    image = cv2.line(image, rect12_start, rect12_end, color, thickness)
    
    image = cv2.line(image, rect21_start, rect21_end, color, thickness)
    image = cv2.line(image, rect22_start, rect22_end, color, thickness)
    
    image = cv2.line(image, rect31_start, rect31_end, color, thickness)
    image = cv2.line(image, rect32_start, rect32_end, color, thickness)
    
    image = cv2.line(image, rect41_start, rect41_end, color, thickness)
    image = cv2.line(image, rect42_start, rect42_end, color, thickness)
    
    image = cv2.line(image, cross1_start, cross1_end, color, thickness)
    image = cv2.line(image, cross2_start, cross2_end, color, thickness)
    
    # add circle
    image = cv2.circle(image, (CENTER_X, CENTER_Y), CIRCLE_DIAMETER, color, thickness)
    
    return image

def draw_distance(image, target_pos=None):
    
    if target_pos is not None:
        color = (255, 0, 0)  # Line color in BGR, e.g., blue
        thickness = 2  # Line thickness in pixels
        
        line_start = (CENTER_X, CENTER_Y)
        line_end = (int(target_pos[0]), int(target_pos[1]))
        
        image = cv2.line(image, line_start, line_end, color, thickness)
        
        # roll
        # roll_start = (int(target_pos[0]), CENTER_Y-15)
        # roll_end = (int(target_pos[0]), CENTER_Y+15)
        # image = cv2.line(image, roll_start, roll_end, color, thickness)
        
        roll_pts = np.array([
            [int(target_pos[0]), CENTER_Y],
            [int(target_pos[0])-10, CENTER_Y+20],
            [int(target_pos[0])+10, CENTER_Y+20]
            ])
        
        cv2.drawContours(image, [roll_pts], 0, color, -1)
        
        dx_end = (int(target_pos[0]), CENTER_Y)
        image = cv2.line(image, (CENTER_X, CENTER_Y), dx_end, color, thickness)
        
        x = (target_pos[0] - CENTER_X)
        image = cv2.putText(image, str(round(x, 2))+' px', (int(target_pos[0])-35, CENTER_Y-17), 
                            cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=color, thickness=2)
        
        # pitch
        # pitch_start = (CENTER_X-15, int(target_pos[1]))
        # pitch_end = (CENTER_X+15, int(target_pos[1]))
        # image = cv2.line(image, pitch_start, pitch_end, color, thickness)
        
        pitch_pts = np.array([
            [CENTER_X, int(target_pos[1])],
            [CENTER_X-20, int(target_pos[1])-10],
            [CENTER_X-20, int(target_pos[1])+10]
            ])
        
        cv2.drawContours(image, [pitch_pts], 0, color, -1)
        
        dy_end = (CENTER_X, int(target_pos[1]))
        image = cv2.line(image, (CENTER_X, CENTER_Y), dy_end, color, thickness)
        
        dy_end = (CENTER_X, int(target_pos[1]))
        image = cv2.line(image, (CENTER_X, CENTER_Y), dy_end, color, thickness)
        
        y = -(target_pos[1] - CENTER_Y)
        image = cv2.putText(image, str(round(y, 2))+' px', (CENTER_X+20, int(target_pos[1])+10), 
                            cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=color, thickness=2)
        
    return image