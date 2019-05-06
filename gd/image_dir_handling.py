import collections
import pathlib
import re
import typing

import log_utils


PerFrameHeatmapDict = typing.Dict[int, log_utils.HeatmapPathDict]


BB_HEIGHT_WIDTH_DIRECTORY_REGEX = re.compile(r"bb_h=([0-9]+)_w=([0-9]+)")
TARGET_IMAGE_DIRECTORY_REGEX = re.compile(r"target_image_([0-9]+)")
FRAME_FILE_REGEX = re.compile(r"([0-9]+)\.[pP][nN][gG]")


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
                if data_frame_number > 32:
                   print(">32 frame #: {}".format(str(heatmap_data_frame_path)))
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
            per_frame_gt_bb_dict[line_index + 1] = log_utils.BoundingBox(center_x=bb_sizes[0] + (bb_sizes[2] / 2) - 1,
                                                                         center_y=bb_sizes[1] + (bb_sizes[3] / 2) - 1,
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
