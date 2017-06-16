import cv2
import numpy as np
from cone_pipeline import ConePipeline
from red_mobile_goal_pipeline import RedMobileGoalPipeline
from blue_mobile_goal_pipeline import BlueMobileGoalPipeline
from PIL import Image
import win32gui
import win32ui
from ctypes import windll

top_list, win_list = [], []


def enum_cb(hwnd, results):
    win_list.append((hwnd, win32gui.GetWindowText(hwnd)))


win32gui.EnumWindows(enum_cb, top_list)
window = [(hwnd, title) for hwnd, title in win_list if 'rvw' in title.lower()]
# just grab the hwnd for first window matching firefox
window = window[0]
print(window)
hwnd = window[0]
win32gui.SetForegroundWindow(hwnd)
hwndDC = win32gui.GetWindowDC(hwnd)
mfcDC = win32ui.CreateDCFromHandle(hwndDC)
saveDC = mfcDC.CreateCompatibleDC()
saveBitMap = win32ui.CreateBitmap()


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


blank_image = cv2.imread('img/blank.png')


def extra_processing_cones(pipeline):
    center_x_positions = []
    center_y_positions = []
    widths = []
    heights = []

    # Find the bounding boxes of the contours to get x, y, width, and height
    for contour in pipeline.filter_contours_output:
        x, y, w, h = cv2.boundingRect(contour)
        center_x_positions.append(x + w / 2)  # X and Y are coordinates of the top-left corner of the bounding box
        center_y_positions.append(y + h / 2)
        widths.append(w)
        heights.append(h)

    # load white image and draw cone contours
    img = blank_image.copy()
    cv2.drawContours(img, pipeline.filter_contours_output, -1, (0, 230, 230), cv2.FILLED)
    return img


def extra_processing_red_mobile_goals(pipeline):
    # transform grayscale to rgb
    out = cv2.cvtColor(pipeline.rgb_threshold_output, cv2.COLOR_GRAY2RGB)

    # invert image so we color the goals
    out = cv2.bitwise_not(out)

    # make red channel 255
    out[:, :, 2] = 255
    return out


def extra_processing_blue_mobile_goals(pipeline):
    # transform grayscale to rgb
    out = cv2.cvtColor(pipeline.rgb_threshold_output, cv2.COLOR_GRAY2RGB)

    # invert image so we color the goals
    out = cv2.bitwise_not(out)

    # make blue channel 255
    out[:, :, 0] = 255
    return out


def main():
    print('Creating pipelines')
    # GRIP generated pipelines for cones and mobile goals
    cone_pipeline = ConePipeline()
    red_mobile_goal_pipeline = RedMobileGoalPipeline()
    blue_mobile_goal_pipeline = BlueMobileGoalPipeline()

    print('Running pipeline')
    while True:
        have_frame, frame = grab_screen()
        if have_frame:
            # GRIP frame processing
            cone_pipeline.process(frame)
            red_mobile_goal_pipeline.process(frame)
            blue_mobile_goal_pipeline.process(frame)

            # Final processing (coloring, etc.)
            images = [frame,
                      extra_processing_cones(cone_pipeline),
                      extra_processing_red_mobile_goals(red_mobile_goal_pipeline),
                      extra_processing_blue_mobile_goals(blue_mobile_goal_pipeline)]

            for i in range(4):
                images[i] = cv2.resize(images[i], None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

            # Show all filters in 4x4 grid
            cv2.imshow('Filters', np.vstack([np.hstack(images[0:2]),
                                             np.hstack(images[2:4])]))
            cv2.waitKey(1)

    # print('Capture closed')
    # cv2.destroyAllWindows()
    # win32gui.DeleteObject(saveBitMap.GetHandle())
    # saveDC.DeleteDC()
    # mfcDC.DeleteDC()
    # win32gui.ReleaseDC(hwnd, hwndDC)


if __name__ == '__main__':
    main()
