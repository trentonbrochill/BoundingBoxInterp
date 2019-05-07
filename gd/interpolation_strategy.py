import copy
import typing

import numpy as np

import log_utils

InterpolationStrategyFunction = typing.Callable[[typing.List[log_utils.LogFrame]], typing.List[log_utils.LogFrame]]

GD_VIDEO_PASS_COUNT = 10
GD_FRAME_PASS_COUNT = 3
GD_ALPHA_XY_AFFINITY = 0.02
GD_ALPHA_HW_AFFINITY = 0.02
GD_ALPHA_ANCHOR = 0.03
SCALE_STEP_SIZE = 5

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 720


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
                                                      start_bb.bottom_right_corner() + (bottom_right_corner_step * index))
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
        log_frames[index].set_bb(bbs[index])

    return log_frames


def _get_affinity_derivative(log_frame: log_utils.LogFrame,
                             start_and_end_frame_numbers: typing.Tuple[int, int],
                             bb_height: log_utils.NumericType,
                             bb_width: log_utils.NumericType,
                             pixel_x: log_utils.NumericType,
                             pixel_y: log_utils.NumericType) -> np.ndarray:
    top_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                       bb_height, bb_width, pixel_x, max(pixel_y - 1, 0))
    bot_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                       bb_height, bb_width, pixel_x, min(pixel_y + 1, IMAGE_HEIGHT))
    left_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                        bb_height, bb_width, max(pixel_x - 1, 0), pixel_y)
    right_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                         bb_height, bb_width, min(pixel_x + 1, IMAGE_WIDTH), pixel_y)

    #print("top_pixel_vals: {}".format(top_pixel_vals))
    #print("bot_pixel_vals: {}".format(bot_pixel_vals))
    #print("left_pixel_vals: {}".format(left_pixel_vals))
    #print("right_pixel_vals: {}".format(right_pixel_vals))

    try:
        taller_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                              bb_height + SCALE_STEP_SIZE, bb_width, pixel_x, pixel_y)
    except KeyError: 
        taller_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                              bb_height, bb_width, pixel_x, pixel_y)

    try:
        shorter_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                               bb_height - SCALE_STEP_SIZE, bb_width, pixel_x, pixel_y)
    except KeyError:
        shorter_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                               bb_height, bb_width, pixel_x, pixel_y)

    try:
        skinnier_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                                bb_height, bb_width - SCALE_STEP_SIZE, pixel_x, pixel_y)
    except KeyError:
        skinnier_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                                bb_height, bb_width, pixel_x, pixel_y)

    try:
        fatter_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                              bb_height, bb_width + SCALE_STEP_SIZE, pixel_x, pixel_y)
    except KeyError:
        fatter_pixel_vals = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                              bb_height, bb_width, pixel_x, pixel_y)

    #print("taller pixels vals: {}".format(taller_pixel_vals))
    #print("shorter pixel vals: {}".format(shorter_pixel_vals))
    #print("fatter_pixel_vals: {}".format(fatter_pixel_vals))
    #print("skinnier_pixel_vals: {}".format(skinnier_pixel_vals))

    #print("x_deriv diff: {}".format(np.array(right_pixel_vals).astype(np.float) - np.array(left_pixel_vals).astype(np.float)))
    #print("y_deriv diff: {}".format(np.array(bot_pixel_vals).astype(np.float) - np.array(top_pixel_vals).astype(np.float)))
    #print("h_deriv diff: {}".format(np.array(taller_pixel_vals).astype(np.float) - np.array(shorter_pixel_vals).astype(np.float)))
    #print("w_deriv diff: {}".format(np.array(fatter_pixel_vals).astype(np.float) - np.array(skinnier_pixel_vals).astype(np.float)))

    x_derivative = np.sum(np.array(right_pixel_vals).astype(np.float) - np.array(left_pixel_vals).astype(np.float))
    y_derivative = np.sum(np.array(bot_pixel_vals).astype(np.float) - np.array(top_pixel_vals).astype(np.float))
    height_derivative = np.sum(np.array(taller_pixel_vals).astype(np.float) - np.array(shorter_pixel_vals).astype(np.float))
    width_derivative = np.sum(np.array(fatter_pixel_vals).astype(np.float) - np.array(skinnier_pixel_vals).astype(np.float))

    #print("x_deriv: {}".format(x_derivative))
    #print("y_deriv: {}".format(y_derivative))
    #print("h_deriv: {}".format(height_derivative))
    #print("w_deriv: {}".format(width_derivative))

    return np.array([x_derivative, y_derivative]), np.array([width_derivative, height_derivative])


def _bound_bb(bb: log_utils.BoundingBox,
              log_frame: log_utils.LogFrame,
              start_and_end_frame_numbers: typing.Tuple[int, int]) -> log_utils.BoundingBox:
    new_bb = copy.deepcopy(bb)

    new_width = 290 if new_bb.get_width() > 290 else new_bb.get_width()
    new_height = 290 if new_bb.get_height() > 290 else new_bb.get_height()

    new_bb = log_utils.BoundingBox(center_x=new_bb.get_center().x,
                                   center_y=new_bb.get_center().y,
                                   width=new_width,
                                   height=new_height)

    while True:
        try:
            _ = log_frame.get_heatmap_data_pixels(start_and_end_frame_numbers,
                                                  new_bb.get_height(), new_bb.get_width(), 0, 0)
            break
        except KeyError as e:
            print(e)
            new_width = new_bb.get_width()
            new_height = new_bb.get_height()
            print("Warning: No heatmap for height {}, width {}, target frames {}, frame {}".format(new_height, new_width, start_and_end_frame_numbers, log_frame.num))
            if new_bb.get_width() < new_bb.get_height():
                new_width += SCALE_STEP_SIZE
            else:
                new_height += SCALE_STEP_SIZE

            if new_width > 295 or new_height > 295:
                raise RuntimeError("Could not find heatmap for target frames {}, frame {}".format(start_and_end_frame_numbers, log_frame.num)) from e

            new_bb = log_utils.BoundingBox(center_x=new_bb.get_center().x,
                                           center_y=new_bb.get_center().y,
                                           width=new_width,
                                           height=new_height)

    if new_bb.left() < 0:
        off_by = abs(new_bb.left())
        new_bb = log_utils.BoundingBox(center_x=new_bb.get_center().x + (off_by / 2),
                                       center_y=new_bb.get_center().y,
                                       width=new_bb.get_width() - off_by,
                                       height=new_bb.get_height())

    if new_bb.right() > IMAGE_WIDTH:
        off_by = new_bb.right() - IMAGE_WIDTH
        new_bb = log_utils.BoundingBox(center_x=new_bb.get_center().x - (off_by / 2),
                                       center_y=new_bb.get_center().y,
                                       width=new_bb.get_width() - off_by,
                                       height=new_bb.get_height())

    if new_bb.top() < 0:
        off_by = abs(new_bb.top())
        new_bb = log_utils.BoundingBox(center_x=new_bb.get_center().x,
                                       center_y=new_bb.get_center().y + (off_by / 2),
                                       width=new_bb.get_width(),
                                       height=new_bb.get_height() - off_by)

    if new_bb.bottom() > IMAGE_HEIGHT:
        off_by = new_bb.bottom() - IMAGE_HEIGHT
        new_bb = log_utils.BoundingBox(center_x=new_bb.get_center().x,
                                       center_y=new_bb.get_center().y - (off_by / 2),
                                       width=new_bb.get_width(),
                                       height=new_bb.get_height() - off_by)

    return new_bb


def gradient_descent_generate_bounding_boxes(log_frames: typing.List[log_utils.LogFrame]) \
        -> typing.List[log_utils.LogFrame]:
    num_frames = len(log_frames)

    # We expect that these were already set to ground truth
    start_bb = log_frames[0].get_bb()
    end_bb = log_frames[-1].get_bb()

    start_and_end_frame_numbers = (log_frames[0].get_number() - 1, log_frames[-1].get_number() - 1)

    # Initialize all of the bounding boxes using linear interpolation
    log_frames = linear_interpolation_generate_bounding_boxes(log_frames)
    
    # Make sure all of our initial bbs are in bounds
    for frame in log_frames[1:-1]:
        frame.set_bb(_bound_bb(frame.get_bb(), frame, start_and_end_frame_numbers))

    for video_pass_index in range(GD_VIDEO_PASS_COUNT):
        # Iterate over all frames between the start and end frame, exclusive
        for frame_index in range(1, num_frames - 1):
            curr_frame = log_frames[frame_index]
            curr_bb = curr_frame.get_bb()
            lin_interp_anchor_bb = _linearly_interpolate_bounding_boxes(start_bb=log_frames[frame_index - 1].get_bb(),
                                                                        end_bb=log_frames[frame_index + 1].get_bb(),
                                                                        num_frames=3)[1]
            #print("curr_frame.get_bb(): {}".format(curr_frame.get_bb()))
            #print("lin_interp_anchor_bb: {}".format(lin_interp_anchor_bb))
            lin_interp_direction = curr_frame.get_bb() - lin_interp_anchor_bb
            #print("lin_interp_direction: {}".format(lin_interp_direction))

            for frame_pass_index in range(GD_FRAME_PASS_COUNT):
                curr_bb = _bound_bb(curr_frame.get_bb(), curr_frame, start_and_end_frame_numbers)
                print("VP{}-FN{}-FP{}: curr_bb={}".format(video_pass_index, frame_index + 1, frame_pass_index,
                    str(curr_bb)))
                xy_affinity_derivative, hw_aff_der = _get_affinity_derivative(log_frame=curr_frame,
                                                               start_and_end_frame_numbers=start_and_end_frame_numbers,
                                                               bb_height=curr_bb.get_height(),
                                                               bb_width=curr_bb.get_width(),
                                                               pixel_x=curr_bb.get_center()[0],
                                                               pixel_y=curr_bb.get_center()[1])

                scaled_xy_aff_der = GD_ALPHA_XY_AFFINITY * xy_affinity_derivative
                scaled_hw_aff_der = GD_ALPHA_HW_AFFINITY * hw_aff_der
                full_aff_der = np.concatenate((scaled_xy_aff_der, scaled_hw_aff_der))
                new_bb = full_aff_der + (GD_ALPHA_ANCHOR * lin_interp_direction) + curr_bb.as_np_array()

                print("VP{}-FN{}-FP{}: {} should be quite a bit less than {} (maybe like a fifth or a tenth)".format(video_pass_index, frame_index + 1, frame_pass_index, 
                    GD_ALPHA_ANCHOR * lin_interp_direction, full_aff_der))
                print("VP{}-FN{}-FP{}: Change in bb: {}".format(video_pass_index, frame_index + 1, frame_pass_index, 
                    new_bb - curr_bb.as_np_array()))

                new_bb_as_obj = log_utils.BoundingBox.from_np_array(new_bb)
                curr_frame.set_bb(_bound_bb(new_bb_as_obj, curr_frame, start_and_end_frame_numbers))

    return log_frames
