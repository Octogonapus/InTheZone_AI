import cv2
import numpy
import math
from enum import Enum


class NumberPipeline:
    """
    An OpenCV pipeline generated by GRIP.
    """

    def __init__(self):
        """initializes all values to presets or None if need to be set
        """

        self.__rgb_threshold_0_red = [192.62589928057554, 231.0665529010239]
        self.__rgb_threshold_0_green = [114.65827338129496, 139.6843003412969]
        self.__rgb_threshold_0_blue = [139.88309352517985, 255.0]

        self.rgb_threshold_0_output = None

        self.__rgb_threshold_1_red = [96.31294964028777, 124.45392491467578]
        self.__rgb_threshold_1_green = [174.28057553956833, 220.1877133105802]
        self.__rgb_threshold_1_blue = [100.89928057553956, 128.80546075085323]

        self.rgb_threshold_1_output = None

        self.__cv_bitwise_or_src1 = self.rgb_threshold_0_output
        self.__cv_bitwise_or_src2 = self.rgb_threshold_1_output

        self.cv_bitwise_or_output = None

    def process(self, source0):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Step RGB_Threshold0:
        self.__rgb_threshold_0_input = source0
        (self.rgb_threshold_0_output) = self.__rgb_threshold(self.__rgb_threshold_0_input, self.__rgb_threshold_0_red,
                                                             self.__rgb_threshold_0_green, self.__rgb_threshold_0_blue)

        # Step RGB_Threshold1:
        self.__rgb_threshold_1_input = source0
        (self.rgb_threshold_1_output) = self.__rgb_threshold(self.__rgb_threshold_1_input, self.__rgb_threshold_1_red,
                                                             self.__rgb_threshold_1_green, self.__rgb_threshold_1_blue)

        # Step CV_bitwise_or0:
        self.__cv_bitwise_or_src1 = self.rgb_threshold_0_output
        self.__cv_bitwise_or_src2 = self.rgb_threshold_1_output
        (self.cv_bitwise_or_output) = self.__cv_bitwise_or(self.__cv_bitwise_or_src1, self.__cv_bitwise_or_src2)

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

    @staticmethod
    def __cv_bitwise_or(src1, src2):
        """Computes the per channel or of two images.
        Args:
            src1: A numpy.ndarray.
            src2: A numpy.ndarray.
        Returns:
            A numpy.ndarray the or of the two mats.
        """
        return cv2.bitwise_or(src1, src2)
