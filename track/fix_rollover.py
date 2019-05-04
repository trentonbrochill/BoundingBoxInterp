import argparse
import os
import re

import cv2
import numpy as np


BB_HEIGHT_WIDTH_DIRECTORY_REGEX = re.compile(r"bb_h=([0-9]+)_w=([0-9]+)")
TARGET_IMAGE_DIRECTORY_REGEX = re.compile(r"target_image_([0-9]+)")
FRAME_FILE_REGEX = re.compile(r"([0-9]+)\.[pP][nN][gG]")
HIGH_DATA_THRESHOLD = 60000
LOW_DATA_THRESHOLD = 5000


def detect_and_fix_rollover(file_path):
    image_data = cv2.imread(file_path)
    last_value = image_data[0]
    found_rollover = True
    ever_found_rollover = False

    while found_rollover:
        found_rollover = False
        for row_index, row in enumerate(image_data):
            start_index = None
            end_index = None
            high_value_before_rollover = None
            for col_index, value in enumerate(row):
                if start_index is None:
                    if last_value > HIGH_DATA_THRESHOLD and value < LOW_DATA_THRESHOLD:
                        start_index = index
                        high_value_before_rollover = last_value
                elif last_value < LOW_DATA_THRESHOLD and value > HIGH_DATA_THRESHOLD:
                    end_index = index
                    found_rollover = True
                    ever_found_rollover = True

            if found_rollover:
                image_data[row_index][start_index:end_index] = high_value_before_rollover

    if ever_found_rollover:
        cv2.imwrite(file_path, image_data)
        print "Fixed rollover in {}".format(file)
    else
        print "Found no rollover in {}".format(file)

    return image_data


def main():
    # base dataset path and setting
    parser = argparse.ArgumentParser(description='Fix rollover by setting it equal to the max val')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video_output_folder', metavar='FOLDER',
                       help='Path of folder to read/write heatmap data (should be root folder containing bb_h=A_w=B folders')
    group.add_argument('--heatmap_data_file', metavar='FILE',
                       help='Specify this instead of the video output folder in order to run on only a single, manually specified heatmap data image')
    args = parser.parse_args()

    if args.video_output_folder is not None:
        abs_video_output_folder = os.path.realpath(args.output_folder)
        print "abs_video_output_folder:", abs_video_output_folder
        for bb_h_w_dir_entry in os.listdir(abs_video_output_folder):
            print "bb_h_w_dir_entry:", bb_h_w_dir_entry
            bb_h_w_dir = os.path.join(abs_dataset_folder, bb_h_w_dir_entry)
            if not os.path.isdir(bb_h_w_dir):
                print "not directory:", bb_h_w_dir
                continue

            if BB_HEIGHT_WIDTH_DIRECTORY_REGEX.search(bb_h_w_dir_entry) is None:
                print "Doesn't match bb Height/Width directory regex:", bb_h_w_dir_entry
                continue

            for target_image_dir_entry in os.listdir(bb_h_w_dir):
                target_image_dir = os.path.join(bb_h_w_dir, target_image_dir_entry)
                if not os.path.isdir(target_image_dir):
                    print "not directory:", target_image_dir
                    continue

                if TARGET_IMAGE_DIRECTORY_REGEX.search(target_image_dir_entry) is None:
                    print "Doesn't match target image directory regex:", target_image_dir_entry
                    continue

                heatmap_data_dir = os.path.join(target_image_dir, "heatmap_data")
                for frame_file_dir_entry in os.listdir(heatmap_data_dir):
                    frame_file = os.path.join(heatmap_data_dir, frame_file_dir_entry)
                    if not os.path.isfile(frame_file):
                        print "not file:", frame_file
                        continue

                    if FRAME_FILE_REGEX.search(frame_file_dir_entry) is None:
                        print "Doesn't match frame file regex:", frame_file_dir_entry
                        continue

                    detect_and_fix_rollover(frame_file)
    else:
        abs_frame_file = os.path.realpath(args.heatmap_data_file)
        detect_and_fix_rollover(abs_frame_file)
                
                
if __name__ == '__main__':
    main()