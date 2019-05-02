import typing

from . import log_utils


InterpolationStrategyFunction = typing.Callable[[typing.List[log_utils.LogFrame]], typing.List[log_utils.LogFrame]]


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


def gradient_descent_generate_bounding_boxes(log_frames: typing.List[log_utils.LogFrame]) \
        -> typing.List[log_utils.LogFrame]:
    num_frames = len(log_frames)

    # We expect that these were already set to ground truth
    start_bb = log_frames[0].get_bb()
    end_bb = log_frames[-1].get_bb()

    # Iterate over all frames between the start and end frame, exclusive
    for index in range(1, num_frames - 1):
        # TODO: TRENTON! Please descend the gradient here.
        pass

    return log_frames