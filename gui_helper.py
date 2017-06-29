import cv2
import numpy as np
from PIL import Image
import win32gui
import win32ui
import win32api
import win32con
import time
from ctypes import windll

top_list, win_list = [], []


def enum_cb(hwnd, results):
    win_list.append((hwnd, win32gui.GetWindowText(hwnd)))


def click(x, y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)


win32gui.EnumWindows(enum_cb, top_list)
window = [(hwnd, title) for hwnd, title in win_list if 'rvw' in title.lower()]
# just grab the hwnd for first window matching firefox
window = window[0]
hwnd = window[0]
win32gui.SetForegroundWindow(hwnd)
hwndDC = win32gui.GetWindowDC(hwnd)
mfcDC = win32ui.CreateDCFromHandle(hwndDC)
saveDC = mfcDC.CreateCompatibleDC()
saveBitMap = win32ui.CreateBitmap()

time.sleep(0.75)  # Sleep to wait for window to be in foreground


def place_waypoints():
    click(1260, 567)  # First waypoint
    time.sleep(0.35)  # Wait for virtual worlds to handle input
    click(1764, 567)  # Second waypoint


def reset_robot():
    click(1170, 1013)  # Press restart
    time.sleep(0.35)  # Wait for virtual worlds to handle input
    click(1775, 1013)  # Press camera angle 2


def grab_screen():
    left, top, right, bot = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bot - top

    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)

    bmp_info = saveBitMap.GetInfo()
    bmp_str = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer('RGB',
                          (bmp_info['bmWidth'], bmp_info['bmHeight']),
                          bmp_str, 'raw', 'BGRX', 0, 1)

    opencv_image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    return result, opencv_image


def cleanup_gui_helper():
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
