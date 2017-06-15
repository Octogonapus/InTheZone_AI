import cv2
import numpy as np
from cone_pipeline import ConePipeline
from red_mobile_goal_pipeline import RedMobileGoalPipeline
from blue_mobile_goal_pipeline import BlueMobileGoalPipeline
from PIL import Image
import win32gui
import win32ui
from ctypes import windll

toplist, winlist = [], []
def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))


win32gui.EnumWindows(enum_cb, toplist)
window = [(hwnd, title) for hwnd, title in winlist if 'rvw' in title.lower()]
# just grab the hwnd for first window matching firefox
window = window[0]
print (window)
hwnd = window[0]

win32gui.SetForegroundWindow(hwnd)

def grab_screen():
    left, top, right, bot = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)
 
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)


    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    return result,opencvImage
    


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
    img = cv2.imread('img/blank.png')
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
   

    print('Creating video capture')
    cap = cv2.VideoCapture(1)

    print('Creating pipelines')
    cone_pipeline = ConePipeline()
    red_mobile_goal_pipeline = RedMobileGoalPipeline()
    blue_mobile_goal_pipeline = BlueMobileGoalPipeline()

    print('Running pipeline')
    while(True):
        have_frame, frame = grab_screen()
        if have_frame:
            cone_pipeline.process(frame)
            red_mobile_goal_pipeline.process(frame)
            blue_mobile_goal_pipeline.process(frame)
            images = [extra_processing_cones(cone_pipeline),
                      extra_processing_red_mobile_goals(red_mobile_goal_pipeline),
                      extra_processing_blue_mobile_goals(blue_mobile_goal_pipeline)]
            #cv2.imshow('frame',frame)
            #cv2.imshow('frame1',images[0])
            #cv2.imshow('frame2',images[1])
            #cv2.imshow('frame3',images[2])


            cv2.imshow('Filters', np.vstack([np.hstack([frame, images[1]]),
                                          np.hstack(images[1:])]))
            cv2.waitKey(1)

    print('Capture closed')
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
