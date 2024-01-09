#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import cv2
import pyautogui
import os
from ADAM.pictotext import pic_to_text


def optics(output_folder="."):
    # Capture a screenshot
    screenshot_path = os.path.join(output_folder, "screenshot.png")
    pyautogui.screenshot(screenshot_path)

    # Check if a camera is available
    camera = cv2.VideoCapture(0)  # 0 represents the default camera (change if necessary)

    if camera.isOpened():
        # Capture a snapshot using the camera
        ret, frame = camera.read()
        if ret:
            snapshot_path = os.path.join(output_folder, "snapshot.png")
            cv2.imwrite(snapshot_path, frame)
            camera.release()  # Release the camera
            screenshot = pic_to_text(screenshot_path)
            snapshot = pic_to_text(snapshot_path)
            return screenshot, snapshot

    # If no camera is available, capture another screenshot as the snapshot
    second_screenshot_path = os.path.join(output_folder, "second_screenshot.png")
    pyautogui.screenshot(second_screenshot_path)
    camera.release()

    screenshot = pic_to_text(screenshot_path)
    snapshot = pic_to_text(second_screenshot_path)
    return screenshot, snapshot