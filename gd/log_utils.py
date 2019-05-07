import os
import pathlib
import typing

import cv2
import numpy as np


NumericType = typing.Union[int, float]


class XYTuple(typing.NamedTuple):
    x: NumericType
    y: NumericType

    def __sub__(self, other):
        if isinstance(other, XYTuple):
            return XYTuple(x=self.x - other.x, y=self.y - other.y)
        else:
            raise TypeError("Subtraction not support between these types: {} - {}".format(str(type(self)),
                                                                                          str(type(other))))

    def __add__(self, other):
        if isinstance(other, XYTuple):
            return XYTuple(x=self.x + other.x, y=self.y + other.y)
        else:
            raise TypeError("Addition not support between these types: {} + {}".format(str(type(self)),
                                                                                       str(type(other))))

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return XYTuple(x=self.x / other, y=self.y / other)
        else:
            raise TypeError("True division not support between these types: {} / {}".format(str(type(self)),
                                                                                            str(type(other))))

    def __floordiv__(self, other):
        if isinstance(other, int):
            return XYTuple(x=self.x // other, y=self.y // other)
        else:
            raise TypeError("Floor division not support between these types: {} // {}".format(str(type(self)),
                                                                                              str(type(other))))

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return XYTuple(x=self.x * other, y=self.y * other)
        else:
            raise TypeError("Multiplication not support between these types: {} * {}".format(str(type(self)),
                                                                                             str(type(other))))


class BoundingBox:
    def __init__(self, center_x: NumericType, center_y: NumericType, width: NumericType, height: NumericType):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height

    @staticmethod
    def from_corners(top_left: XYTuple,
                     top_right: XYTuple,
                     bottom_left: XYTuple,
                     bottom_right: XYTuple):
        center_x = round((top_left.x + top_right.x) / 2)
        center_y = round((top_left.y + bottom_left.y) / 2)
        width = bottom_right.x - bottom_left.x
        height = bottom_right.y - top_right.y 
        return BoundingBox(center_x, center_y, width, height)

    def top_left_corner(self) -> XYTuple:
        return XYTuple(x=self.center_x - (self.width / 2),
                       y=self.center_y - (self.height / 2))

    def top_right_corner(self) -> XYTuple:
        return XYTuple(x=self.center_x + (self.width / 2),
                       y=self.center_y - (self.height / 2))

    def bottom_right_corner(self) -> XYTuple:
        return XYTuple(x=self.center_x + (self.width / 2),
                       y=self.center_y + (self.height / 2))

    def bottom_left_corner(self) -> XYTuple:
        return XYTuple(x=self.center_x - (self.width / 2),
                       y=self.center_y + (self.height / 2))

    def top(self) -> NumericType:
        return self.center_y - (self.height / 2)

    def left(self) -> NumericType:
        return self.center_x - (self.width / 2)

    def right(self) -> NumericType:
        return self.center_x + (self.width / 2)

    def bottom(self) -> NumericType:
        return self.center_y + (self.height / 2)

    def get_all_corners(self) -> typing.Tuple[XYTuple, XYTuple, XYTuple, XYTuple]:
        # top left, top right, bottom right, bottom left
        return self.top_left_corner(), self.top_right_corner(), self.bottom_right_corner(), self.bottom_left_corner()

    def get_center(self) -> XYTuple:
        return XYTuple(x=self.center_x,
                       y=self.center_y)

    def get_width(self) -> NumericType:
        return self.width

    def get_height(self) -> NumericType:
        return self.height

    def to_list(self) -> typing.List[NumericType]:
        return [self.center_x, self.center_y, self.width, self.height]

    def as_np_array(self) -> np.ndarray:
        return np.array(self.to_list())

    @staticmethod
    def from_np_array(bb_as_np_array) -> 'BoundingBox':
        return BoundingBox(*list(bb_as_np_array))

    @staticmethod
    def from_list(bb_as_list) -> 'BoundingBox':
        return BoundingBox(*bb_as_list)

    def __sub__(self, other):
        if isinstance(other, BoundingBox):
            return np.array([self.center_x - other.center_x,
                             self.center_y - other.center_y,
                             self.width - other.width,
                             self.height - other.height])
        else:
            raise TypeError("Subtraction not support between these types: {} - {}".format(str(type(self)),
                                                                                          str(type(other))))

    def __str__(self):
        return "BoundingBox[Center=({:.2f}, {:.2f});HxW=({:.2f}x{:.2f})".format(self.center_x, self.center_y, self.height, self.width)


class HeatmapConfiguration(typing.NamedTuple):
    bb_height: int
    bb_width: int
    target_frame: int


HeatmapPathDict = typing.Dict[HeatmapConfiguration, pathlib.Path]


class LogFrame:
    def __init__(self,
                 frame_number: int,
                 image_path: pathlib.Path,
                 heatmap_dict: HeatmapPathDict,
                 ground_truth_bounding_box: BoundingBox):
        self.num = frame_number
        self.image_path = image_path
        self.heatmap_path_dict = heatmap_dict
        self.gt_bb = ground_truth_bounding_box
        self.bb = None  # type: typing.Union[BoundingBox, None]
        self.bb_is_ground_truth = False
        self.heatmap_data_cache = {}



    def _get_heatmap_with_config(self, heatmap_config: HeatmapConfiguration) -> np.ndarray:
        try:
            self.heatmap_path_dict[heatmap_config].resolve(strict=True)
        except FileNotFoundError as e:
            print("os.exists: {}".format(os.exists(str(self.heatmap_path_dict[heatmap_config]))))
            raise KeyError("No heatmap for frame {}, config {}, expected path {}".format(self.num, str(heatmap_config), str(self.heatmap_path_dict[heatmap_config]))) from e

        return cv2.imread(str(self.heatmap_path_dict[heatmap_config]), cv2.IMREAD_ANYDEPTH)

    def get_image(self) -> np.ndarray:
        return cv2.imread(str(self.image_path))

    @staticmethod
    def round_to_multiple_of_five(numeric_value: NumericType) -> int:
        return 5 * int(round(numeric_value / 5))

    def _get_cached_heatmap(self, bb_height: NumericType, bb_width: NumericType, target_frame: int) -> np.ndarray:
        rounded_bb_height = LogFrame.round_to_multiple_of_five(bb_height)
        rounded_bb_width = LogFrame.round_to_multiple_of_five(bb_width)
        heatmap_config = HeatmapConfiguration(bb_height=rounded_bb_height,
                                              bb_width=rounded_bb_width,
                                              target_frame=target_frame)
 
        if heatmap_config not in self.heatmap_data_cache.keys():
            self.heatmap_data_cache[heatmap_config] = self._get_heatmap_with_config(heatmap_config)

        return self.heatmap_data_cache[heatmap_config]


    def get_heatmap_data_pixels(self,
                                start_and_end_frame_numbers: typing.Tuple[int, int],
                                bb_height: NumericType,
                                bb_width: NumericType,
                                pixel_x: NumericType,
                                pixel_y: NumericType) -> typing.Tuple[np.uint16, np.uint16]:
        rounded_pixel_x = int(round(pixel_x))
        rounded_pixel_y = int(round(pixel_y))
        return (self._get_cached_heatmap(bb_height, bb_width, start_and_end_frame_numbers[0])[rounded_pixel_y][rounded_pixel_x],
                self._get_cached_heatmap(bb_height, bb_width, start_and_end_frame_numbers[1])[rounded_pixel_y][rounded_pixel_x])

    def clear_heatmap_data_cache(self):
        del self.heatmap_data_cache
        self.heatmap_data_cache = {}

    def get_number(self) -> int:
        return self.num

    def set_bb(self, estimated_bb: BoundingBox):
        assert not self.bb_is_ground_truth, "A frame that is set to ground truth should not be given a bb estimate"
        self.bb = estimated_bb

    def get_bb(self) -> BoundingBox:
        return self.bb

    def get_gt_bb(self) -> BoundingBox:
        return self.gt_bb

    def set_to_ground_truth(self):
        if not self.bb_is_ground_truth:
            self.bb = self.gt_bb
            self.bb_is_ground_truth = True

    def is_set_to_ground_truth(self) -> bool:
        return self.bb_is_ground_truth

    def clear_bb(self):
        self.bb = None
        self.bb_is_ground_truth = False

    def bb_error(self) -> float:
        """Calculate intersection over union error between the estimated and ground truth bounding boxes."""
        assert self.bb is not None, "Cannot calculate error on a frame whose bounding box has not been set"
        assert self.gt_bb is not None, "Frame's ground truth bounding box is None"

        # Determine the (x, y)-coordinates of the intersection rectangle
        intersect_right = min(self.bb.right(), self.gt_bb.right())
        intersect_left = max(self.bb.left(), self.gt_bb.left())
        intersect_top = max(self.bb.top(), self.gt_bb.top())
        intersect_bottom = min(self.bb.bottom(), self.gt_bb.bottom())

        # Compute the area of intersection rectangle
        intersect_area = max(0, intersect_right - intersect_left + 1) * max(0, intersect_bottom - intersect_top  + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        gt_bb_area = self.gt_bb.get_height() * self.gt_bb.get_width()
        bb_area = self.bb.get_height() * self.bb.get_width()

        # Compute the intersection over union by taking the intersection area and dividing it by the sum of
        # prediction + ground-truth areas - the intersection area
        iou = intersect_area / float(gt_bb_area + bb_area - intersect_area)

        # Return the error based on the intersection over union value
        return 1 - iou

    def is_correct(self, error_under_this_is_correct: NumericType) -> bool:
        return self.bb_error() <= error_under_this_is_correct

    def to_json_dict(self) -> dict:
        return {"frame_num": self.num, "bb": self.bb.to_list(), "bb_err": self.bb_error()}

                 


