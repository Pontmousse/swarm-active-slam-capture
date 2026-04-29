######################################################################################
######################################################################################
######################################################################################
######################################################################################

# Read me instructions for screen recording video

# Open a terminal prompt in current folder
# Type command: python3 Animate_Mapping.py
# Open 'Recording.py' in VS code
# Edit number of frames in line 53 to determine the length of the screen video record
# Run 'Recording.py' then execute terminal command asap
# Watch animation run (you can increze window size, play with the Open3D display)
# Wait until pop-up window shows recording has finished
# close animation window
# search for 'animation.mp4' file with recorded video in current folder

######################################################################################
######################################################################################
######################################################################################
######################################################################################

import open3d as o3d
import pyautogui
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Function to capture screenshots of the rendering window
def capture_screenshots(num_frames, frame_rate):
    screenshots = []
    for i in range(num_frames):
        print(f'Capturing Screenshots...Frame {i}/{num_frames}')
        # screenshot = pyautogui.screenshot(region=(0, 0, 1440, 900))  # Adjust the region to fit your rendering window
        screenshot = pyautogui.screenshot()
        
        screenshots.append(screenshot)
        time.sleep(frame_rate)  # Adjust the interval based on your desired frame rate
    return screenshots

# Function to convert screenshots to a video
def create_video(num_frames, screenshots, output_video):
    first_screenshot = screenshots[0]
    width, height = first_screenshot.size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 10, (width, height))  # Adjust the frame rate
    for i, screenshot in enumerate(screenshots):
        print(f'Recording......Frame {i}/{num_frames}')
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()


if __name__ == "__main__":
    # Capture screenshots
    num_frames = 500  # Adjust as needed
    frame_rate = 0.1
    screenshots = capture_screenshots(num_frames, frame_rate)
    
    # Create video
    output_dir = Path(__file__).resolve().parent / "Outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = str(output_dir / "animation.mp4")
    print('\n')
    create_video(num_frames, screenshots, output_video)

    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo("Recording", "Recording Finished")

    # root.mainloop()
