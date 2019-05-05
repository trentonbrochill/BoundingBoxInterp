import typing

from . import log_utils


InterpolationStrategyFunction = typing.Callable[[typing.List[log_utils.LogFrame]], typing.List[log_utils.LogFrame]]


GRADIENT_DESCENT_VIDEO_PASS_COUNT = 10



def linear_interpolation_generate_bounding_boxes(log_frames: typing.List[log_utils.LogFrame]) \
        -> typing.List[log_utils.LogFrame]:
    num_frames = len(log_frames)

    # We expect that these were already set to ground truth
    start_bb = log_frames[0].get_bb()
    end_bb = log_frames[-1].get_bb()

    top_left_corner_step = (end_bb.top_left_corner() - start_bb.top_left_corner()) / num_frames
    top_right_corner_step = (end_bb.top_right_corner() - start_bb.top_right_corner()) / num_frames
    bottom_left_corner_step = (end_bb.bottom_left_corner() - start_bb.bottom_left_corner()) / num_frames
    bottom_right_corner_step = (end_bb.bottom_right_corner() - start_bb.bottom_right_corner()) / num_frames

    # Iterate over all frames between the start and end frame, exclusive
    for index in range(1, num_frames - 1):
        frame_bb = log_utils.BoundingBox.from_corners(start_bb.top_left_corner() + (top_left_corner_step * index),
                                                      start_bb.top_right_corner() + (top_right_corner_step * index),
                                                      start_bb.bottom_left_corner() + (bottom_left_corner_step * index),
                                                      start_bb.bottom_right_corner() + (bottom_right_corner_step * index))
        log_frames[index].set_bb(frame_bb)

    return log_frames


def get_affinity_der_pos(log_frame: log_utils.LogFrame,
                         start_and_end_frame_numbers: typing.Tuple[int, int],
                         bb_height: log_utils.NumericType,
                         bb_width: log_utils.NumericType,
                         pixel_x: log_utils.NumericType,
                         pixel_y: log_utils.NumericType) -> typing.Tuple[float, float]:
    top_pixel_vals = log_frame.
    
    


def gradient_descent_generate_bounding_boxes(log_frames: typing.List[log_utils.LogFrame]) \
        -> typing.List[log_utils.LogFrame]:
    num_frames = len(log_frames)

    # We expect that these were already set to ground truth
    start_bb = log_frames[0].get_bb()
    end_bb = log_frames[-1].get_bb()

    for passIndex in range(pass
    # Iterate over all frames between the start and end frame, exclusive
    for index in range(1, num_frames - 1):
        

    return log_frames
