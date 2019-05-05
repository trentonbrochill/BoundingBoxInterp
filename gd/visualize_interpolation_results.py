import argparse
import json
import pathlib

import cv2
import numpy as np

import log_utils
import image_dir_handling


def visualize_frame(all_frames, interp_type_short, num_adj, frame):
    image = all_frames[frame["frame_num"] - 1].get_image()
    gt_bb = all_frames[frame["frame_num"] - 1].get_gt_bb()
    bb = log_utils.BoundingBox(*frame["bb"])

    im_show = image  #cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.rectangle(im_show, bb.top_left_corner(), bb.bottom_right_corner(), (0, 255, 0), 3)
    cv2.rectangle(im_show, gt_bb.top_left_corner(), gt_bb.bottom_right_corner(), (0, 0, 255), 3)
    cv2.putText(im_show,
                interp_type_short + "_f{}_{}-adjs".format(frame["frame_num"], num_adj),
                (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.CV_AA)
    cv2.imshow("{}_results".format(interp_type_short), im_show)
    cv2.waitKey(50)

def main():
    parser = argparse.ArgumentParser(description='Visualizes a bb_results_<time>.json file')
    parser.add_argument('bb_results_file', metavar='FILE_PATH', required=True,
                        help='The path to the bb_results_<time>.json file to be visualized')
    args = parser.parse_args()

    json_file_path = pathlib.Path(args.bb_results_file).resolve(strict=True)

    with open(str(json_file_path), 'r') as json_file:
        json_data = json.load(json_file)

    video_input_folder_path = pathlib.Path(json_data["video_input_folder"]).resolve(strict=True)
    video_output_folder_path = pathlib.Path(json_data["video_output_folder"]).resolve(strict=True)

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

    interpolation_types = [("lin", "linear_interpolation"), ("gd", "gradient_descent_interpolation")]
    for interp_type_short, interp_type_json_key in interpolation_types:
        iterations_list = json_data[interp_type_json_key]["iterations"]
        for iteration in iterations_list:
            num_adj = iteration["total_number_of_adjustments"]
            iter_frames_list = iteration["all_frames_estimated_bbs"]
            for frame in iter_frames_list:
                visualize_frame(all_frames, interp_type_short, num_adj, frame)




if __name__ == "__main__":
    main()
