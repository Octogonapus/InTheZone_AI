from vjoy import vj, set_joy
from mpmath import *
import cv2
import numpy as np
import time

from cone_pipeline import ConePipeline
from red_mobile_goal_pipeline import RedMobileGoalPipeline
from blue_mobile_goal_pipeline import BlueMobileGoalPipeline
from number_pipeline import NumberPipeline

from heap_scanner import HeapScanner
from gui_helper import click, place_waypoints, reset_robot, grab_screen, cleanup_gui_helper, start_stop_program

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
    # blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = 1
    params.blobColor = 255
    params.minThreshold = 10
    params.maxThreshold = 220
    params.filterByArea = True
    params.minArea = 190
    params.filterByCircularity = True
    params.minCircularity = 0.25
    params.maxCircularity = 10000.0
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)

    # filter blobs by position
    blobs = detector.detect(cv2.medianBlur(pipeline.rgb_threshold_output, 3))
    for blob in reversed(blobs):  # do it in reverse because we are removing elements as we iterate
        coord = blob.pt
        if coord[0] > 730 and coord[1] < 25:
            # blob is top right box
            blobs.remove(blob)
        elif coord[0] < 70 and coord[1] > 560:
            # blob is bottom left box
            blobs.remove(blob)

    # draw blobs
    img = blank_image.copy()
    img = cv2.drawKeypoints(pipeline.rgb_threshold_output, blobs, img,
                            (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img


def extra_processing_blue_mobile_goals(pipeline):
    # blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = 1
    params.blobColor = 255
    params.minThreshold = 10
    params.maxThreshold = 220
    params.filterByArea = True
    params.minArea = 190
    params.filterByCircularity = True
    params.minCircularity = 0.25
    params.maxCircularity = 10000.0
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)

    # filter blobs by position
    blobs = detector.detect(cv2.medianBlur(pipeline.rgb_threshold_output, 3))
    for blob in reversed(blobs):  # do it in reverse because we are removing elements as we iterate
        coord = blob.pt
        if coord[0] > 730 and coord[1] < 25:
            # blob is top right box
            blobs.remove(blob)
        elif coord[0] < 70 and coord[1] > 560:
            # blob is bottom left box
            blobs.remove(blob)

    # draw blobs
    img = blank_image.copy()
    img = cv2.drawKeypoints(pipeline.rgb_threshold_output, blobs, img,
                            (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img


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
    print("vJoy Opening")
    vj.open()
    time.sleep(0.5)

    # place_waypoints()

    print("Scanning Memory")
    scanner = HeapScanner(0x03600000)
    wp1d_addr = scanner.scan_memory("2.1 m")
    wp1a_addr = scanner.scan_memory("-54.91")
    wp2a_addr = scanner.scan_memory("+99.91")

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

            r = num_from_distance_string(scanner.read_memory(wp1d_addr, 6))
            theta = num_from_angle_string(scanner.read_memory(wp2a_addr, 6))
            wp1_t = num_from_angle_string(scanner.read_memory(wp1a_addr, 6))
            if r is not None and theta is not None and wp1_t is not None:
                theta = radians(theta)
                x = r * fabs(mp.cos(theta))
                y = x * mp.tan(theta)
                print("x:", x, "y:", y, "t:", mp.atan2(y, x) - wp1_t)
                print(r, theta)
            else:
                print("none :(")

            # To write to joystick call set_joy
            # set_joy(xval, yval)

            # Show all filters in 4x4 grid
            cv2.imshow('Filters', np.hstack([np.vstack([np.hstack(images[0:2]),
                                                        np.hstack(images[2:4])]),
                                             np.vstack([images[4],
                                                        blank_image_scaled])]))
            if cv2.waitKey(10) == 27:
                print('Capture closed')
                cv2.destroyAllWindows()
                cleanup_gui_helper()
                vj.close()
                return


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
                return -1 * float(text[1:len(text) - 1])
            elif text.startswith("+"):
                _last_sign = 1
                return float(text[1:len(text) - 1])
        except ValueError:
            print("num_from_angle_string error: ", text)


if __name__ == '__main__':
    main()
