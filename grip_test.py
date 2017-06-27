import cv2
import numpy as np
from PIL import Image
import win32gui
import win32ui
from ctypes import windll
from cone_pipeline import ConePipeline
from red_mobile_goal_pipeline import RedMobileGoalPipeline
from blue_mobile_goal_pipeline import BlueMobileGoalPipeline
from number_pipeline import NumberPipeline
from heap_scanner import HeapScanner

top_list, win_list = [], []


def enum_cb(hwnd, results):
    win_list.append((hwnd, win32gui.GetWindowText(hwnd)))


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
blank_image_scaled = cv2.resize(blank_image,
                                None,
                                fx=0.6,
                                fy=0.6,
                                interpolation=cv2.INTER_AREA)


def extra_processing_cones(pipeline):
    center_x_positions = []
    center_y_positions = []
    widths = []
    heights = []

    # Find the bounding boxes of the contours to get x, y, width, and height
    for contour in pipeline.filter_contours_output:
        x, y, w, h = cv2.boundingRect(contour)
        # X and Y are coordinates of the top-left corner of the bounding box
        center_x_positions.append(x + w / 2)
        center_y_positions.append(y + h / 2)
        widths.append(w)
        heights.append(h)

    # load white image and draw cone contours
    img = blank_image.copy()
    cv2.drawContours(img,
                     pipeline.filter_contours_output,
                     -1,
                     (0, 230, 230),
                     cv2.FILLED)
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


number_template_names = ['img/number_pattern/zero.png',
                         'img/number_pattern/one.png',
                         'img/number_pattern/two.png',
                         'img/number_pattern/three.png',
                         'img/number_pattern/four.png',
                         'img/number_pattern/five.png',
                         'img/number_pattern/six.png',
                         'img/number_pattern/seven.png',
                         'img/number_pattern/eight.png',
                         'img/number_pattern/nine.png']
number_templates = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_RGB2GRAY) for file in number_template_names]
num_temp_dims = [temp.shape for temp in number_templates]


def extra_processing_numbers(pipeline):
    # Scale image for easier OCR
    image = pipeline.cv_bitwise_or_output.copy()
    # scaled_source = cv2.resize(pipeline.cv_bitwise_or_output, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)

    # Pattern matching
    for i in range(len(number_templates)):
        match = cv2.matchTemplate(pipeline.cv_bitwise_or_output, number_templates[i], cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
        top_left = max_loc
        bottom_right = (top_left[0] + num_temp_dims[i][0], top_left[1] + num_temp_dims[i][1])
        cv2.rectangle(image, top_left, bottom_right, 255, 2)

    # transform grayscale to rgb
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def main():
    print("Scanning Memory")
    scanner = HeapScanner(0x03340000)
    wp1d_addr = scanner.scan_memory("2.11 m")
    wp1a_addr = scanner.scan_memory("-54.84")
    wp2d_addr = scanner.scan_memory("3.69 m")
    wp2a_addr = scanner.scan_memory("+99.68")

    print('Creating pipelines')
    # GRIP generated pipelines for cones and mobile goals
    cone_pipeline = ConePipeline()
    rmg_pipeline = RedMobileGoalPipeline()
    bmg_pipeline = BlueMobileGoalPipeline()
    number_pipeline = NumberPipeline()

    print('Running pipeline')
    while True:
        have_frame, frame = grab_screen()
        if have_frame:
            # GRIP frame processing
            cone_pipeline.process(frame)
            rmg_pipeline.process(frame)
            bmg_pipeline.process(frame)
            number_pipeline.process(frame)

            # Final processing (coloring, etc.)
            images = [frame,
                      extra_processing_cones(cone_pipeline),
                      extra_processing_red_mobile_goals(rmg_pipeline),
                      extra_processing_blue_mobile_goals(bmg_pipeline),
                      extra_processing_numbers(number_pipeline)]

            for i in range(len(images)):
                images[i] = cv2.resize(images[i], None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

            read = scanner.read_memory(wp1a_addr, 6)
            # print(num_from_distance_string(read))
            print(num_from_angle_string(read))
            # print(read)
            # print([int(s) for s in read.split() if s.isdigit()])

            # Show all filters in 4x4 grid
            cv2.imshow('Filters', np.hstack([np.vstack([np.hstack(images[0:2]),
                                                        np.hstack(images[2:4])]),
                                             np.vstack([images[4],
                                                        blank_image_scaled])]))
            cv2.waitKey(1)

            # print('Capture closed')
            # cv2.destroyAllWindows()
            # win32gui.DeleteObject(saveBitMap.GetHandle())
            # saveDC.DeleteDC()
            # mfcDC.DeleteDC()
            # win32gui.ReleaseDC(hwnd, hwndDC)


def num_from_distance_string(text):
    try:
        if text.endswith(" "):
            if len(text) == 1:
                return int(text[0])
            else:
                return float(text[0:3])
        elif text.endswith("m"):
            return float(text[0:3])
        else:
            return float(text)
    except ValueError:
        print("num_from_distance_string error: ", text)


_last_sign = 1


def num_from_angle_string(text):
    global _last_sign

    try:
        if text.startswith("-"):
            _last_sign = -1
            return -1 * float(text[1:])
        elif text.startswith("+"):
            _last_sign = 1
            return float(text[1:])
        else:
            return _last_sign * float(text[1:])
    except ValueError:
        try:
            if text.startswith("-"):
                _last_sign = -1
                return -1 * float(text[1:len(text)-1])
            elif text.startswith("+"):
                _last_sign = 1
                return float(text[1:len(text)-1])
        except ValueError:
            print("num_from_angle_string error: ", text)


if __name__ == '__main__':
    main()
