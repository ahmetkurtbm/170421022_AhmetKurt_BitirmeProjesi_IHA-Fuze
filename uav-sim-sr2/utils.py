import pyautogui
import time

def switch_tab():
    pyautogui.hotkey('alt', 'tab')
    
def unpause_game():
    pyautogui.press('p')
    
def relaunch(end_flight=[117, 131], retry_undo=[513, 586], retry=[509, 525]):
    pyautogui.press('esc')
    time.sleep(0.5)
    pyautogui.click(x=end_flight[0], y=end_flight[1])
    time.sleep(0.5)
    pyautogui.click(x=retry_undo[0], y=retry_undo[1])
    time.sleep(0.5)
    pyautogui.click(x=retry[0], y=retry[1])
    
def open_navball(position=[718, 923]):
    pyautogui.click(x=position[0], y=position[1])
    
def switch_to_fpv():
    pyautogui.press('c')
    time.sleep(0.1)
    pyautogui.press('c')