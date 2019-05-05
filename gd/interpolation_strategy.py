import typing

import numpy as np

from . import log_utils

InterpolationStrategyFunction = typing.Callable[[typing.List[log_utils.LogFrame]], typing.List[log_utils.LogFrame]]

GD_VIDEO_PASS_COUNT = 10
GD_FRAME_PASS_COUNT = 3
GD_ALPHA_AFFINITY = 0.001
GD_ALPHA_ANCHOR = 0.1
SCALE_STEP_SIZE = 5


def _linearly_interpolate_bounding_boxes(start_bb: log_utils.BoundingBox,
                                         end_bb: log_utils.BoundingBox,
                                         num_frames: int) -> typing.List[log_utils.BoundingBox]:
    top_left_corner_step = (end_bb.top_left_corner() - start_bb.top_left_corner()) / num_frames
    top_right_corner_step = (end_bb.top_right_corner() - start_bb.top_right_corner()) / num_frames
    bottom_left_corner_step = (end_bb.bottom_left_corner() - start_bb.bottom_left_corner()) / num_frames
    bottom_right_corner_step = (end_bb.bottom_right_corner() - start_bb.bottom_right_corner()) / num_frames

    # Iterate over all frames between the start and end frame, exclusive
    bbs = [start_bb]
    for index in range(1, num_frames - 1):
        frame_bb = log_utils.BoundingBox.from_corners(start_bb.top_left_corner() + (top_left_corner_step * index),
                                                      start_bb.top_right_corner() + (top_right_corner_step * index),
                                                      start_bb.bottom_left_corner() + (bottom_left_corner_step * index),
                                                      start_bb.bottom_right_corner() + (
                                                                  bottom_right_corner_step * index))
        bbs.append(frame_bb)

    bbs.append(end_bb)
    return bbs


def linear_interpolation_generate_bounding_boxes(log_frames: typing.List[log_utils.LogFrame]) \
        -> typing.List[log_utils.LogFrame]:
    num_frames = len(log_frames)

    # We expect that these were already set to ground truth
    start_bb = log_frames[0].get_bb()
    end_bb = log_frames[-1].get_bb()

    bbs = _linearly_interpolate_bounding_boxes(start_bb, end_bb, num_frames)

    # Iterate over all frames between the start and end frame, exclusive
    for index in range(1, num_frames - 1):
        log_frames[index].set_bb(bbs[index - 1])

    return log_frames


def _get_affinity_derivative(log_frame: log_utils.LogFrame,
                             start_and_end_frame_numbers: typing.Tuple[int, int],
                             bb_height: log_utils.NumericType,
                             bb_width: log_utils.NumericType,
                             pixel_x: log_utils.NumericType,
                             pixel_y: log_utils.NumericType) -> np.ndarray:
    top_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                       bb_height, bb_width, pixel_x, pixel_y + 1)
    bot_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                       bb_height, bb_width, pixel_x, pixel_y - 1)
    left_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                        bb_height, bb_width, pixel_x - 1, pixel_y)
    right_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                         bb_height, bb_width, pixel_x + 1, pixel_y)
    taller_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                          bb_height + SCALE_STEP_SIZE, bb_width, pixel_x, pixel_y)
    shorter_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                           bb_height - SCALE_STEP_SIZE, bb_width, pixel_x, pixel_y)
    skinnier_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                            bb_height, bb_width - SCALE_STEP_SIZE, pixel_x, pixel_y)
    fatter_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                          bb_height, bb_width + SCALE_STEP_SIZE, pixel_x, pixel_y)

    x_derivative = np.sum(np.array(right_pixel_vals) - np.array(left_pixel_vals))
    y_derivative = np.sum(np.array(top_pixel_vals) - np.array(bot_pixel_vals))
    height_derivative = np.sum(np.array(taller_pixel_vals) - np.array(shorter_pixel_vals))
    width_derivative = np.sum(np.array(fatter_pixel_vals) - np.array(skinnier_pixel_vals))

    return np.array([x_derivative[0], y_derivative[0], width_derivative[0], height_derivative[0]])


def gradient_descent_generate_bounding_boxes(log_frames: typing.List[log_utils.LogFrame]) \
        -> typing.List[log_utils.LogFrame]:
    num_frames = len(log_frames)

    # We expect that these were already set to ground truth
    start_bb = log_frames[0].get_bb()
    end_bb = log_frames[-1].get_bb()

    start_and_end_frame_numbers = (log_frames[0].get_number(), log_frames[-1].get_number())

    # Initialize all of the bounding boxes using linear interpolation
    log_frames = linear_interpolation_generate_bounding_boxes(log_frames)

    for _ in range(GD_VIDEO_PASS_COUNT):
        # Iterate over all frames between the start and end frame, exclusive
        for frame_index in range(1, num_frames - 1):
            curr_frame = log_frames[frame_index]
            curr_bb = curr_frame.get_bb()
            lin_interp_anchor_bb = _linearly_interpolate_bounding_boxes(start_bb=log_frames[frame_index - 1].get_bb(),
                                                                        end_bb=log_frames[frame_index + 1].get_bb(),
                                                                        num_frames=3)[1]
            lin_interp_direction = curr_frame.get_bb() - lin_interp_anchor_bb

            for _ in range(GD_FRAME_PASS_COUNT):
                curr_bb = curr_frame.get_bb()
                affinity_derivative = _get_affinity_derivative(log_frame=curr_frame,
                                                               start_and_end_frame_numbers=start_and_end_frame_numbers,
                                                               bb_height=curr_bb.get_height(),
                                                               bb_width=curr_bb.get_width(),
                                                               pixel_x=curr_bb.get_center()[0],
                                                               pixel_y=curr_bb.get_center()[1])

                new_bb = (GD_ALPHA_AFFINITY * affinity_derivative) + (GD_ALPHA_ANCHOR * lin_interp_direction) \
                         + curr_bb.as_np_array()

                curr_frame.set_bb(new_bb)

    return log_frames
