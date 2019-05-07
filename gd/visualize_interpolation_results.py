import argparse
import base64
import collections
import copy
import hashlib
import json
import os
import pathlib

import cv2
import numpy as np

import log_utils
import image_dir_handling


def round_corner(corner):
    return int(round(corner[0])), int(round(corner[1]))


def visualize_frame(all_frames, interp_type_short, num_adj, frame):
    image = all_frames[frame["frame_num"] - 1].get_image()
    gt_bb = all_frames[frame["frame_num"] - 1].get_gt_bb()
    bb = log_utils.BoundingBox(*frame["bb"])

    im_show = image  #cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.rectangle(im_show, round_corner(bb.top_left_corner()), round_corner(bb.bottom_right_corner()), (0, 255, 0), 3)
    #cv2.rectangle(im_show, round_corner(gt_bb.top_left_corner()), round_corner(gt_bb.bottom_right_corner()), (0, 0, 255), 3)
    cv2.putText(im_show,
                interp_type_short + "_f{}_{}-adjs".format(frame["frame_num"], num_adj),
                (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("{}_results".format(interp_type_short), im_show)
    cv2.waitKey(133)

def main():
    parser = argparse.ArgumentParser(description='Visualizes a bb_results_<time>.json file')
    parser.add_argument('bb_results_file', metavar='FILE_PATH',
                        help='The path to the bb_results_<time>.json file to be visualized')
    args = parser.parse_args()

    json_file_path = pathlib.Path(args.bb_results_file).resolve(strict=True)

    with open(str(json_file_path), 'r') as json_file:
        json_data = json.load(json_file)

    video_input_folder_path = pathlib.Path(json_data["video_input_folder"]).resolve(strict=True)
    video_output_folder_path = pathlib.Path(json_data["video_output_folder"]).resolve(strict=True)

    in_folder_hash = hashlib.sha1("{}".format(video_input_folder_path).encode('utf-8'))
    out_folder_hash = hashlib.sha1("{}".format(video_output_folder_path).encode('utf-8'))
    json_file_name = "{}_{}.json".format(base64.urlsafe_b64encode(in_folder_hash.digest()[:20]).decode("utf-8") ,
                                         base64.urlsafe_b64encode(out_folder_hash.digest()[:20]).decode("utf-8") )

    json_file_path = pathlib.Path(os.path.join(os.getcwd(), json_file_name)).resolve()
    if json_file_path.is_file():
        with open(str(json_file_path), 'r') as json_file:
            json_dict = json.load(json_file)
            per_frame_gt_bb_dict_from_json = json_dict["per_frame_gt_bb_dict"]
            per_frame_image_dict_from_json = json_dict["per_frame_image_dict"]
            per_frame_heatmap_dict_from_json = json_dict["per_frame_heatmap_dict"]

        per_frame_gt_bb_dict = {}
        for key, value in per_frame_gt_bb_dict_from_json.items():
            per_frame_gt_bb_dict[int(key)] = log_utils.BoundingBox.from_list(value)

        per_frame_image_dict = {}
        for key, value in per_frame_image_dict_from_json.items():
            per_frame_image_dict[int(key)] = pathlib.Path(value)

        per_frame_heatmap_dict = collections.defaultdict(dict)
        for key, value in per_frame_heatmap_dict_from_json.items():
            for key2, value2 in value.items():
                key2_tokens = key2.split(',')
                key2_config = log_utils.HeatmapConfiguration(bb_height=int(key2_tokens[0]),
                                                             bb_width=int(key2_tokens[1]),
                                                             target_frame=int(key2_tokens[2]))
                per_frame_heatmap_dict[int(key)][key2_config] = pathlib.Path(value2)
    else:
        groundtruth_rect_file_path = video_input_folder_path.joinpath("groundtruth_rect.txt")
        input_video_dir_path = video_input_folder_path.joinpath("img")

        per_frame_gt_bb_dict = image_dir_handling.get_per_frame_groundtruth_bbs(groundtruth_rect_file_path)
        per_frame_image_dict = image_dir_handling.get_per_frame_image_paths(input_video_dir_path)
        per_frame_heatmap_dict = image_dir_handling.get_per_frame_heatmap_dictionaries(video_output_folder_path)

        gt_bb_dict_copy = copy.deepcopy(per_frame_gt_bb_dict)
        for key, value in gt_bb_dict_copy.items():
            gt_bb_dict_copy[key] = value.to_list()

        per_frame_image_dict_copy = copy.deepcopy(per_frame_image_dict)
        for key, value in per_frame_image_dict_copy.items():
            per_frame_image_dict_copy[key] = str(value)

        per_frame_heatmap_dict_copy = collections.defaultdict(dict)
        for key, value in per_frame_heatmap_dict.items():
            for key2, value2 in value.items():
                key2_str = "{},{},{}".format(key2.bb_height, key2.bb_width, key2.target_frame)
                per_frame_heatmap_dict_copy[key][key2_str] = str(value2)

        with open(str(json_file_path), 'w') as json_file:
            json_dict = {}
            json_dict["per_frame_gt_bb_dict"] = gt_bb_dict_copy
            json_dict["per_frame_image_dict"] = per_frame_image_dict_copy
            json_dict["per_frame_heatmap_dict"] = per_frame_heatmap_dict_copy
            json.dump(json_dict, json_file, indent='  ')
    assert sorted(per_frame_gt_bb_dict.keys()) == sorted(per_frame_heatmap_dict.keys()), \
        "Expect same frames of heatmap data and ground truth bound box data"

    all_frames = []
    for frame_number in sorted(per_frame_gt_bb_dict.keys()):
        all_frames.append(log_utils.LogFrame(frame_number=frame_number,
                                             image_path=per_frame_image_dict[frame_number],
                                             heatmap_dict=per_frame_heatmap_dict[frame_number],
                                             ground_truth_bounding_box=per_frame_gt_bb_dict[frame_number]))

    interpolation_types = [("lin", "linear_interpolation"), ("gd", "gradient_descent_interpolation")]
    for interp_type_short, interp_type_json_key in interpolation_types:
        iterations_list = json_data[interp_type_json_key]["iterations"]
        for iteration in iterations_list:
            num_adj = iteration["number_of_adjustments"]
            iter_frames_list = iteration["all_frames_estimated_bbs"]
            for frame in iter_frames_list:
                visualize_frame(all_frames, interp_type_short, num_adj, frame)




if __name__ == "__main__":
    main()
