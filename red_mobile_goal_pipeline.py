import cv2
import numpy
import math
from enum import Enum


class RedMobileGoalPipeline:
    """
    An OpenCV pipeline generated by GRIP.
    """

    def __init__(self):
        """initializes all values to presets or None if need to be set
        """

        self.__rgb_threshold_red = [4.586330935251798, 255.0]
        self.__rgb_threshold_green = [0.0, 0.0]
        self.__rgb_threshold_blue = [0.0, 0.0]

        self.rgb_threshold_output = None

    def process(self, source0):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Step RGB_Threshold0:
        self.__rgb_threshold_input = source0
        (self.rgb_threshold_output) = self.__rgb_threshold(self.__rgb_threshold_input, self.__rgb_threshold_red,
                                                           self.__rgb_threshold_green, self.__rgb_threshold_blue)

    @staticmethod
    def __rgb_threshold(input, red, green, blue):
        """Segment an image based on color ranges.
        Args:
            input: A BGR numpy.ndarray.
            red: A list of two numbers the are the min and max red.
            green: A list of two numbers the are the min and max green.
            blue: A list of two numbers the are the min and max blue.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        return cv2.inRange(out, (red[0], green[0], blue[0]), (red[1], green[1], blue[1]))
