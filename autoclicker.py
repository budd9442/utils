import pyautogui
import time

try:
    while True:
        pyautogui.click()
        time.sleep(1)
except KeyboardInterrupt:
    print("Autoclicker stopped.")