#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import time
import typing

from . import log_utils
from . import interpolation_strategy
from . import image_dir_handling


# Error is (1 - IOU)
MOST_ERROR_THRESHOLD_FOR_RECURSION_EXIT = 0.1


total_number_of_adjustments = 0


def generate_bb_and_determine_num_adjustments(all_frames: typing.Sequence[log_utils.LogFrame],
                                              split_depth: int,
                                              json_dict: dict,
                                              frames_in_use: typing.Sequence[log_utils.LogFrame],
                                              interp_strat: interpolation_strategy.InterpolationStrategyFunction) -> int:
    global total_number_of_adjustments

    # Set the first and last frames to use ground truth
    frames_in_use[0].set_to_ground_truth()
    frames_in_use[-1].set_to_ground_truth()

    # Get list of BBs using the given interpolation strategy
    frames_with_bbs = interp_strat(frames_in_use)

    # Find frame with most erroneous bb
    most_error = -1
    most_erroneous_index = -1
    per_image_error = []
    for index, frame in enumerate(frames_with_bbs):
        this_frame_error = frame.bb_error()
        per_image_error.append(per_image_error)
        if this_frame_error > most_error:
            most_error = this_frame_error
            most_erroneous_index = index

    # Store the intermediate results after this iteration
    this_iteration_json_dict = {"split_depth": split_depth,
                                "total_number_of_adjustments": total_number_of_adjustments,
                                "all_frames_estimated_bbs": []}
    for frame in all_frames:
        this_iteration_json_dict["all_frames_estimated_bbs"].append(frame.to_json_dict())

    if "iterations" not in json_dict:
        json_dict["iterations"] = []
    json_dict["iterations"].append(this_iteration_json_dict)

    # If most erroneous error < threshold, return (bbs, 0)
    if most_error < MOST_ERROR_THRESHOLD_FOR_RECURSION_EXIT:
        return 0

    # The "labeler" needs to adjust the most erroneous frame
    num_adjustments = 1
    total_number_of_adjustments += 1

    # Otherwise, set bb of most erroneous frame to ground truth bb for that frame
    frames_with_bbs[most_erroneous_index].set_to_ground_truth()

    # Now split and redo interpolation on either side
    num_adjustments += generate_bb_and_determine_num_adjustments(all_frames=all_frames,
                                                                 split_depth=split_depth + 1,
                                                                 json_dict=json_dict,
                                                                 frames_in_use=frames_with_bbs[0:most_erroneous_index + 1],
                                                                 interp_strat=interp_strat)
    num_adjustments += generate_bb_and_determine_num_adjustments(all_frames=all_frames,
                                                                 split_depth=split_depth + 1,
                                                                 json_dict=json_dict,
                                                                 frames_in_use=frames_with_bbs[most_erroneous_index:-1],
                                                                 interp_strat=interp_strat)

    # Return the total number of adjustments that needed to be done for this set of frames
    return num_adjustments


def main():
    global total_number_of_adjustments

    parser = argparse.ArgumentParser(description='Generate bounding boxes using gradient descent '
                                                 'or linear interpolation')
    parser.add_argument('--video_input_folder', metavar='FOLDER', required=True,
                        help='Path to the folder that contains the input images and the groundtruth_rect.txt file for'
                             'the video being used')
    parser.add_argument('--video_output_folder', metavar='FOLDER', required=True,
                        help='Path to the folder containing the generated heatmaps for the video being used')
    parser.add_argument('--estimated_bb_output_folder', metavar='FOLDER', required=False, default=os.getcwd(),
                        help='Path to put the generated file containing the list of estimated bounding boxes')

    args = parser.parse_args()

    # TODO: We'll probably end up wanting a way to visualize the generated bounding boxes, but that's easy because we
    #  can just copy the code that does that in DCFNet.py

    video_input_folder_path = pathlib.Path(args.groundtruth_rect).resolve(strict=True)
    video_output_folder_path = pathlib.Path(args.video_output_folder).resolve(strict=True)
    output_folder_path = pathlib.Path(args.estimated_bb_output_folder).resolve(strict=True)

    groundtruth_rect_file_path = video_input_folder_path.joinpath("groundtruth_rect.txt")
    input_video_dir_path = video_input_folder_path.joinpath("img")

    per_frame_gt_bb_dict = image_dir_handling.get_per_frame_groundtruth_bbs(groundtruth_rect_file_path)
    per_frame_image_dict = image_dir_handling.get_per_frame_image_paths(input_video_dir_path)
    per_frame_heatmap_dict = image_dir_handling.get_per_frame_heatmap_dictionaries(video_output_folder_path)

    assert sorted(per_frame_gt_bb_dict.keys()) == sorted(per_frame_heatmap_dict.keys()), \
        "Expect same frames of heatmap data and ground truth bound box data"

    all_frames = []
    for frame_number in sorted(per_frame_gt_bb_dict.keys()):
        all_frames.append(log_utils.LogFrame(frame_number=frame_number,
                                             image_path=per_frame_image_dict[frame_number],
                                             heatmap_dict=per_frame_heatmap_dict[frame_number],
                                             ground_truth_bounding_box=per_frame_gt_bb_dict[frame_number]))

    output_json_dict = {
        "video_input_folder": str(video_input_folder_path),
        "video_output_folder": str(video_output_folder_path),
        "linear_interpolation": {},
        "gradient_descent_interpolation": {}
    }

    # Do linear interpolation first
    total_number_of_adjustments = 0
    lin_splits = generate_bb_and_determine_num_adjustments(all_frames=all_frames,
                                                           split_depth=0,
                                                           json_dict=output_json_dict["linear_interpolation"],
                                                           frames_in_use=all_frames,
                                                           interp_strat=interpolation_strategy.linear_interpolation_generate_bounding_boxes)

    output_json_dict["linear_interpolation"]["total_num_splits"] = lin_splits

    # Reset the bounding boxes of all frames
    for frame in all_frames:
        frame.clear_bb()

    # Then do gradient descent interpolation
    total_number_of_adjustments = 0
    gd_splits = generate_bb_and_determine_num_adjustments(all_frames=all_frames,
                                                          split_depth=0,
                                                          json_dict=output_json_dict["gradient_descent_interpolation"],
                                                          frames_in_use=all_frames,
                                                          interp_strat=interpolation_strategy.gradient_descent_generate_bounding_boxes)

    output_json_dict["gradient_descent_interpolation"]["total_num_splits"] = gd_splits

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_json_dict["start_time"] = timestr

    output_file_path = output_folder_path.joinpath("bb_results_" + timestr + ".json")
    with open(str(output_file_path), 'w') as out_file:
        json.dump(output_json_dict, out_file, indent="  ")


if __name__ == "__main__":
    main()
