import open3d as o3d
import pyautogui
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Function to capture screenshots of the rendering window
def capture_screenshots(num_frames, frame_rate):
    screenshots = []
    for i in range(num_frames):
        screenshot = pyautogui.screenshot(region=(0, 0, 1440, 900))  # Adjust the region to fit your rendering window
        screenshots.append(screenshot)
        print(f'Screenshot {i} / {num_frames} captured')
        time.sleep(frame_rate)  # Adjust the interval based on your desired frame rate
    return screenshots

# Function to convert screenshots to a video
def create_video(screenshots, output_video):
    first_screenshot = screenshots[0]
    width, height = first_screenshot.size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 10, (width, height))  # Adjust the frame rate
    i = 1
    for screenshot in screenshots:
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        video.write(frame)
        print(f'Frame {i} / {num_frames} written')
        i += 1
    video.release()


# Capture screenshots
num_frames = 100  # Adjust as needed
frame_rate = 0.1
screenshots = capture_screenshots(num_frames, frame_rate)
    
# Create video
output_video = "animation.mp4"
print('Initialization')
create_video(screenshots, output_video)

root = tk.Tk()
root.withdraw()

messagebox.showinfo("Recording", "Recording Finished")

root.mainloop()
