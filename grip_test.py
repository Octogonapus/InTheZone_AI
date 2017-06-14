import cv2
from networktables import NetworkTables
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

    # Publish to the '/vision/red_areas' network table
    table = NetworkTables.getTable('/vision/red_areas')
    table.putNumberArray('x', center_x_positions)
    table.putNumberArray('y', center_y_positions)
    table.putNumberArray('width', widths)
    table.putNumberArray('height', heights)

    # load white image and draw cone contours
    img = cv2.imread('img/blank.png')
    cv2.drawContours(img, pipeline.filter_contours_output, -1, (0, 230, 230), 2)
    return img


def extra_processing_red_mobile_goals(pipeline):
    out = cv2.cvtColor(pipeline.rgb_threshold_output, cv2.COLOR_GRAY2RGB)
    out[:, :, 2] = 255
    return out


def extra_processing_blue_mobile_goals(pipeline):
    out = cv2.cvtColor(pipeline.rgb_threshold_output, cv2.COLOR_GRAY2RGB)
    out[:, :, 0] = 255
    return out


def main():
    print('Initializing NetworkTables')
    NetworkTables.setClientMode()
    NetworkTables.setIPAddress('localhost')
    NetworkTables.initialize()

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
            cv2.imshow('Filters', np.column_stack(images))
            cv2.waitKey(1)
            # for i in range(3):
            #     print(images[i].shape)

    print('Capture closed')


if __name__ == '__main__':
    main()
