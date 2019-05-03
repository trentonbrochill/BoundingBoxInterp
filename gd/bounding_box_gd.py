#!/usr/bin/env python3

import argparse
import typing

from . import log_utils
from . import interpolation_strategy


# Error is (1 - IOU)
MOST_ERROR_THRESHOLD_FOR_RECURSION_EXIT = 0.1


def generate_bb_and_determine_num_adjustments(frames: typing.Sequence[log_utils.LogFrame],
                                              interp_strat: interpolation_strategy.InterpolationStrategyFunction) -> int:
    # Set the first and last frames to use ground truth
    frames[0].set_to_ground_truth()
    frames[-1].set_to_ground_truth()

    # Get list of BBs using the given interpolation strategy
    frames_with_bbs = interp_strat(frames)

    # Find frame with most erroneous bb
    most_error = -1
    most_erroneous_index = -1
    for index, frame in enumerate(frames_with_bbs):
        this_frame_error = frame.bb_error()
        if this_frame_error > most_error:
            most_error = this_frame_error
            most_erroneous_index = index

    # If most erroneous error < threshold, return (bbs, 0)
    if most_error < MOST_ERROR_THRESHOLD_FOR_RECURSION_EXIT:
        return 0

    # The "labeler" needs to adjust the most erroneous frame
    num_adjustments = 1

    # Otherwise, set bb of most erroneous frame to ground truth bb for that frame
    frames_with_bbs[most_erroneous_index].set_to_ground_truth()

    # Now split and redo interpolation on either side
    num_adjustments += generate_bb_and_determine_num_adjustments(frames_with_bbs[0:most_erroneous_index + 1],
                                                                 interp_strat)
    num_adjustments += generate_bb_and_determine_num_adjustments(frames_with_bbs[most_erroneous_index:-1],
                                                                 interp_strat)

    # Return the total number of adjustments that needed to be done for this set of frames
    return num_adjustments


def main():
    parser = argparse.ArgumentParser(description='Generate bounding boxes using gradient descent '
                                                 'or linear interpolation')
    parser.add_argument('--dataset_folder', metavar='FOLDER', required=True,
                        help='Path of folder containing all dataset video folders')
    parser.add_argument('--output_folder', metavar='FOLDER', required=True,
                        help='Path of folder to write all generated heatmaps for videos in dataset_folder')
    args = parser.parse_args()

    # TODO: Properly set up command line arguments, populate initial list of log_utils.LogFrame's, then make the initial
    #  call to generate_bb_and_determine_num_adjustments(...). We'll probably end up wanting a way to visualize the
    #  generated bounding boxes, but that's easy because we can just copy the code that does that in DCFNet.py


if __name__ == "__main__":
    main()
