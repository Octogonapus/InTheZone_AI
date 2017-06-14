import cv2
import numpy as np
from cone_pipeline import ConePipeline
from red_mobile_goal_pipeline import RedMobileGoalPipeline
from blue_mobile_goal_pipeline import BlueMobileGoalPipeline


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
    while cap.isOpened():
        have_frame, frame = cap.read()
        if have_frame:
            cone_pipeline.process(frame)
            red_mobile_goal_pipeline.process(frame)
            blue_mobile_goal_pipeline.process(frame)
            images = [extra_processing_cones(cone_pipeline),
                      extra_processing_red_mobile_goals(red_mobile_goal_pipeline),
                      extra_processing_blue_mobile_goals(blue_mobile_goal_pipeline)]
            cv2.imshow('Filters', np.vstack([np.hstack([frame, images[0]]),
                                             np.hstack(images[1:])]))
            cv2.waitKey(1)

    print('Capture closed')


if __name__ == '__main__':
    main()
