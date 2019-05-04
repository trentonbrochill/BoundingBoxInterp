#!/usr/bin/env python3

import argparse
import collections
import json
import os
import pathlib
import re
import time
import typing

from . import log_utils
from . import interpolation_strategy


PerFrameHeatmapDict = typing.Dict[int, log_utils.HeatmapPathDict]


# Error is (1 - IOU)
MOST_ERROR_THRESHOLD_FOR_RECURSION_EXIT = 0.1
BB_HEIGHT_WIDTH_DIRECTORY_REGEX = re.compile(r"bb_h=([0-9]+)_w=([0-9]+)")
TARGET_IMAGE_DIRECTORY_REGEX = re.compile(r"target_image_([0-9]+)")
FRAME_FILE_REGEX = re.compile(r"([0-9]+)\.[pP][nN][gG]")


def generate_bb_and_determine_num_adjustments(all_frames: typing.Sequence[log_utils.LogFrame],
                                              this_iteration_number: int,
                                              json_dict: dict,
                                              frames_in_use: typing.Sequence[log_utils.LogFrame],
                                              interp_strat: interpolation_strategy.InterpolationStrategyFunction) -> int:
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
    this_iteration_json_dict = {"iteration_number": this_iteration_number, "all_frames_estimated_bbs": []}
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

    # Otherwise, set bb of most erroneous frame to ground truth bb for that frame
    frames_with_bbs[most_erroneous_index].set_to_ground_truth()

    # Now split and redo interpolation on either side
    num_adjustments += generate_bb_and_determine_num_adjustments(all_frames=all_frames,
                                                                 this_iteration_number=this_iteration_number + 1,
                                                                 json_dict=json_dict,
                                                                 frames_in_use=frames_with_bbs[0:most_erroneous_index + 1],
                                                                 interp_strat=interp_strat)
    num_adjustments += generate_bb_and_determine_num_adjustments(all_frames=all_frames,
                                                                 this_iteration_number=this_iteration_number + 1,
                                                                 json_dict=json_dict,
                                                                 frames_in_use=frames_with_bbs[most_erroneous_index:-1],
                                                                 interp_strat=interp_strat)

    # Return the total number of adjustments that needed to be done for this set of frames
    return num_adjustments


def get_per_frame_heatmap_dictionaries(video_output_folder_path: pathlib.Path) -> PerFrameHeatmapDict:
    per_frame_heatmap_dict = collections.defaultdict(dict)
    for bb_h_w_dir_entry in video_output_folder_path.iterdir():
        if not bb_h_w_dir_entry.is_dir():
            print("\"{}\" is not a directory; skipping...".format(str(bb_h_w_dir_entry)))
            continue

        expected_bb_h_w_directory = bb_h_w_dir_entry.name
        bb_h_w_dir_match = BB_HEIGHT_WIDTH_DIRECTORY_REGEX.fullmatch(expected_bb_h_w_directory)
        if bb_h_w_dir_match is None:
            print("\"{}\" (basename of \"{}\") does not match the "
                  "BB height/width directory regex".format(expected_bb_h_w_directory, str(bb_h_w_dir_entry)))
            continue

        bb_h = int(bb_h_w_dir_match.group(1))
        bb_w = int(bb_h_w_dir_match.group(2))

        for target_image_dir_entry in bb_h_w_dir_entry.iterdir():
            if not target_image_dir_entry.is_dir():
                print("\"{}\" is not a directory; skipping...".format(str(target_image_dir_entry)))
                continue

            expected_taget_image_directory = target_image_dir_entry.name
            target_image_match = TARGET_IMAGE_DIRECTORY_REGEX.fullmatch(expected_taget_image_directory)
            if target_image_match is None:
                print("\"{}\" (basename of \"{}\") does not match the "
                      "target image directory regex".format(expected_taget_image_directory,
                                                            str(target_image_dir_entry)))
                continue

            target_image = int(target_image_match.group(1))

            heatmap_config = log_utils.HeatmapConfiguration(bb_height=bb_h,
                                                            bb_width=bb_w,
                                                            target_frame=target_image)

            heatmap_data_dir = target_image_dir_entry.joinpath("heatmap_data").resolve(strict=True)
            for heatmap_data_frame_path in heatmap_data_dir.iterdir():
                if not heatmap_data_frame_path.is_file():
                    print("\"{}\" is not a file; skipping...".format(str(heatmap_data_frame_path)))
                    continue

                expected_data_frame_name = heatmap_data_frame_path.name
                data_frame_match = FRAME_FILE_REGEX.fullmatch(expected_data_frame_name)
                if data_frame_match is None:
                    print("\"{}\" (basename of \"{}\") does not match the "
                          "frame file regex".format(expected_data_frame_name,
                                                    str(heatmap_data_frame_path)))
                    continue

                data_frame_number_string = data_frame_match.group(1)
                data_frame_number = int(data_frame_number_string.lstrip("0"))
                per_frame_heatmap_dict[data_frame_number][heatmap_config] = heatmap_data_frame_path

    return per_frame_heatmap_dict


def get_per_frame_groundtruth_bbs(groundtruth_rect_path: pathlib.Path) -> typing.Dict[int, log_utils.BoundingBox]:
    # Read the ground truth bounding boxes for each image in this video
    per_frame_gt_bb_dict = {}

    with open(str(groundtruth_rect_path), 'r') as groundtruth_file:
        all_lines = groundtruth_file.readlines()

        # Make sure sure all_lines is defined and is not empty
        if not all_lines:
            raise RuntimeError("{} is empty".format(str(groundtruth_rect_path)))

        for line_index, groundtruth_line in enumerate(all_lines):
            list_of_bb_size_strings = groundtruth_line.strip().replace(' ', '').split(',')
            assert len(list_of_bb_size_strings) == 4, "Expect 4 "
            bb_sizes = [float(size_str) for size_str in list_of_bb_size_strings]
            per_frame_gt_bb_dict[line_index + 1] = log_utils.BoundingBox(center_x=bb_sizes[0],
                                                                         center_y=bb_sizes[1],
                                                                         width=bb_sizes[2],
                                                                         height=bb_sizes[3])

    return per_frame_gt_bb_dict


def get_per_frame_image_paths(input_image_dir_path: pathlib.Path) -> typing.Dict[int, pathlib.Path]:
    per_frame_image_path_dict = {}

    for frame_path in input_image_dir_path.iterdir():
        if not frame_path.is_file():
            print("\"{}\" is not a file; skipping...".format(str(frame_path)))
            continue

        expected_frame_name = frame_path.name
        frame_match = FRAME_FILE_REGEX.fullmatch(expected_frame_name)
        if frame_match is None:
            print("\"{}\" (basename of \"{}\") does not match the "
                  "frame file regex".format(expected_frame_name, str(frame_path)))
            continue

        frame_number_string = frame_match.group(1)
        frame_number = int(frame_number_string.lstrip("0"))
        per_frame_image_path_dict[frame_number] = frame_path

    return per_frame_image_path_dict


def main():
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

    per_frame_gt_bb_dict = get_per_frame_groundtruth_bbs(groundtruth_rect_file_path)
    per_frame_image_dict = get_per_frame_image_paths(input_video_dir_path)
    per_frame_heatmap_dict = get_per_frame_heatmap_dictionaries(video_output_folder_path)

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
    lin_splits = generate_bb_and_determine_num_adjustments(all_frames=all_frames,
                                                           this_iteration_number=0,
                                                           json_dict=output_json_dict["linear_interpolation"],
                                                           frames_in_use=all_frames,
                                                           interp_strat=interpolation_strategy.linear_interpolation_generate_bounding_boxes)

    output_json_dict["linear_interpolation"]["total_num_splits"] = lin_splits

    # Reset the bounding boxes of all frames
    for frame in all_frames:
        frame.clear_bb()

    # Then do gradient descent interpolation
    gd_splits = generate_bb_and_determine_num_adjustments(all_frames=all_frames,
                                                          this_iteration_number=0,
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
