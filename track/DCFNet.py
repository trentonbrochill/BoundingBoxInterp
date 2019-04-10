from os.path import join, isdir
from os import makedirs
import os
import sys
import argparse
import json
import numpy as np
import errno
import torch

import math
import cv2
import matplotlib.pyplot as plt
import time as time
from util import *
from net import DCFNet
from eval_otb import eval_auc

SKIPPED_RETURN_VALUE = "Skipped"
SUCCESS_RETURN_VALUE = "Success"

def apply_matplotlib_colormap(image, cmap=plt.get_cmap('viridis')):

    assert image.dtype == np.uint8, 'must be np.uint8 image'
    if image.ndim == 3: image = image.squeeze(-1)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:,0:3]    # color range RGBA => RGB
    color_range = (color_range*255.0).astype(np.uint8)         # [0,1] => [0,255]
    color_range = np.squeeze(np.dstack([color_range[:,2], color_range[:,1], color_range[:,0]]), 0)  # RGB => BGR

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image, color_range[:,i]) for i in range(3)]
    return np.dstack(channels)


class TrackerConfig(object):
    # These are the default hyper-params for DCFNet
    # OTB2013 / AUC(0.665)
    feature_path = 'param.pth'
    scale_factors = np.array([.5, 1])
    #crop_sz = (int(math.ceil(720*scale_factors[1])), int(math.ceil(480*scale_factors[0])) )
    square_crop_side_size = 1440
    crop_sz = (square_crop_side_size, square_crop_side_size)

    lambda0 = 1e-4
    padding = 1.5
    output_sigma_factor = 0.1
    interp_factor = 0.01
    axis_num_scale = 1
    num_scale = axis_num_scale ** 2
    scale_step = 1.0275  # 1.0275
    scale_factor = scale_step ** (np.arange(axis_num_scale) - axis_num_scale / 2)
    scale_factor_pairs = [np.array((scale_factor[i / axis_num_scale], scale_factor[i % axis_num_scale]))
                          for i in range(num_scale)]



    scale_step ** (np.arange(axis_num_scale) - axis_num_scale / 2)
    min_scale_factor = 0.2
    max_scale_factor = 5
    scale_penalty = 0.95
    axis_scale_penalties = scale_penalty ** (np.abs((np.arange(axis_num_scale) - axis_num_scale / 2)))
    scale_penalties = [(axis_scale_penalties[i / axis_num_scale] + axis_scale_penalties[i % axis_num_scale]) / 2
                       for i in range(num_scale)]

    net_input_size = [crop_sz[0], crop_sz[1]]
    net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
    output_sigma = max(crop_sz) / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, net_input_size)
    yf = torch.rfft(torch.Tensor(y).view(1, 1, crop_sz[1], crop_sz[0]).cuda(), signal_ndim=2)
    cos_window = torch.Tensor(np.outer(np.hanning(crop_sz[1]), np.hanning(crop_sz[0]))).cuda()

    def __init__(self, square_crop_size_side=720):
        self.square_crop_side_size = square_crop_size_side
        self.crop_sz = (self.square_crop_side_size, self.square_crop_side_size)

        self.net_input_size = [self.crop_sz[0], self.crop_sz[1]]
        self.net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
        self.output_sigma = 900 / (1 + self.padding) * self.output_sigma_factor
        #self.output_sigma = max(self.crop_sz) / (1 + self.padding) * self.output_sigma_factor
        self.y = gaussian_shaped_labels(self.output_sigma, self.net_input_size)
        cv2.imshow("gaussian_shaped_labels", self.y)
        self.yf = torch.rfft(torch.Tensor(self.y).view(1, 1, self.crop_sz[1], self.crop_sz[0]).cuda(), signal_ndim=2)
        #print "np.outer(np.hanning(self.crop_sz[0]), np.hanning(self.crop_sz[1])).shape:", np.outer(np.hanning(self.crop_sz[0]), np.hanning(self.crop_sz[1])).shape
        #input()
        self.cos_window = torch.Tensor(np.outer(np.hanning(self.crop_sz[1]), np.hanning(self.crop_sz[0]))).cuda()


class DCFNetTraker(object):
    def __init__(self, im, init_rect, config=TrackerConfig(), gpu=True):
        self.gpu = gpu
        self.config = config
        self.net = DCFNet(config)
        self.net.load_param(config.feature_path)
        self.net.eval()
        if gpu:
            self.net.cuda()

        # confine results
        target_pos, target_sz = rect1_2_cxy_wh(init_rect)
        self.min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
        self.max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

        # crop template
        window_sz = target_sz * (1 + config.padding)
        bbox = cxy_wh_2_bbox(target_pos, window_sz)
        patch = crop_chw(im, bbox, self.config.crop_sz)

        target = patch - config.net_average_image
        self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())
        self.target_pos, self.target_sz = target_pos, target_sz
        self.patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)  # buff

    def track(self, im):
        for i in range(self.config.num_scale):  # crop multi-scale search region
            window_sz = self.target_sz * (self.config.scale_factor[i] * (1 + self.config.padding))
            bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
            self.patch_crop[i, :] = crop_chw(im, bbox, self.config.crop_sz)

        search = self.patch_crop - self.config.net_average_image

        if self.gpu:
            response = self.net(torch.Tensor(search).cuda())
        else:
            response = self.net(torch.Tensor(search))
        peak, idx = torch.max(response.view(self.config.num_scale, -1), 1)
        peak = peak.data.cpu().numpy() * self.config.scale_penalties
        best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx[best_scale], self.config.net_input_size)

        if r_max > self.config.net_input_size[0] / 2:
            r_max = r_max - self.config.net_input_size[0]
        if c_max > self.config.net_input_size[1] / 2:
            c_max = c_max - self.config.net_input_size[1]
        window_sz = self.target_sz * (self.config.scale_factor[best_scale] * (1 + self.config.padding))

        self.target_pos = self.target_pos + np.array([c_max, r_max]) * window_sz / self.config.net_input_size
        self.target_sz = np.minimum(np.maximum(window_sz / (1 + self.config.padding), self.min_sz), self.max_sz)

        # model update
        window_sz = self.target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
        patch = crop_chw(im, bbox, self.config.crop_sz)
        target = patch - self.config.net_average_image
        self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=self.config.interp_factor)

        return cxy_wh_2_rect1(self.target_pos, self.target_sz)  # 1-index


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def rearrangeMolecules(im):
    im_h = im.shape[0]
    im_w = im.shape[1]
    return np.vstack([np.hstack([im[im_h/2:im_h,im_w/2:im_w,:],im[im_h/2:im_h,0:im_w/2,:]]),np.hstack([im[0:im_h/2,im_w/2:im_w,:],im[0:im_h/2,0:im_w/2,:]])])


def normalize_to_255(im):
    return (((im * 2.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)


def normalize_to_uint16(im):
    return (((im * 2.0) + 1.0) / 2.0 * 65535.0).astype(np.uint16)


def to_heatmap(im):
    return apply_matplotlib_colormap(im )


def generate_heatmap_for_specific_target_and_scale(input_video_folder, num_images, scale_factor_pair, 
                                                   target_image_index, target_groundtruth_bb, output_folder):
    #scale_factor_pair = [1,1]

    use_gpu = True
    visualization = False

    lowest_min = 100
    highest_max = -100

    init_rect = np.array(target_groundtruth_bb).astype(np.float)
    image_files = [os.path.join(input_video_folder, "img", "{:04}.png".format(img_num + 1)) 
                   for img_num in range(num_images)]
    n_images = num_images

    tic = time.time()  # time start

    target_pos, target_sz = rect1_2_cxy_wh(init_rect)  # OTB label is 1-indexed

    im = cv2.imread(image_files[target_image_index])  # HxWxC

    # Create default parameters structure
    

    # Determine the appropriate output square size of the patch for this scale factor pair
    output_square_size = int(max(scale_factor_pair[1] * im.shape[1], scale_factor_pair[0] * im.shape[0]))

    if output_square_size >= 1300:
        print "Skipping scale factor pair {} because it requires too large of a scaled image".format(scale_factor_pair)
        return SKIPPED_RETURN_VALUE
    else:
        print "Generating heatmap for scale factor pair {} and target index {}".format(scale_factor_pair,
                                                                                       target_image_index)

    output_folder_heatmap_dir = os.path.join(output_folder, "heatmap_images")
    output_folder_data_dir = os.path.join(output_folder, "heatmap_data")
    mkdir_p(output_folder_heatmap_dir)
    mkdir_p(output_folder_data_dir)

    # load feature extractor network
    config = TrackerConfig(output_square_size)
    net = DCFNet(config)
    net.load_param(args.model)
    net.eval().cuda()
   
    #print image_files[0]
    #input()
    # confine results
    min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
    max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

    # crop template
    #print im.shape
    #input()
    window_sz = target_sz * (1 + config.padding)
    bbox = cxy_wh_2_bbox(target_pos, window_sz)
    patch = crop_chw(im, bbox, (300, 300),[104, 117, 123])
    patch = pad_to_size_centered(patch, config.crop_sz,[104, 117, 123])
    #patch = resize_with_pad_to_square(im, config.crop_sz)
    #transposed_patch = np.array([s.transpose(1,2,0) / 255.0 for s in patch_crop])
    if visualization:
        cv2.imshow("patch", (patch.transpose(1,2,0) / 255.0))
        cv2.waitKey(500)
        print patch.shape

    target = patch - config.net_average_image
    net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())

    #print config.scale_penalties
    #np.set_printoptions(linewidth=120)
    #print np.array(config.scale_penalties).reshape((9,9))
    #input()
    speed = []

    res = [cxy_wh_2_rect1(target_pos, target_sz)]  # save in .txt
    patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)
    for f in range(n_images):  # track
        # Skip the target index
        #if f == target_image_index:
        #    continue

        im = cv2.imread(image_files[f])
        #print config.scale_factor

        #for i in range(config.num_scale):  # crop multi-scale search region
            #x_scale_index = i / config.axis_num_scale  # Integer division
            #y_scale_index = i % config.axis_num_scale

            #window_sz = target_sz * (config.scale_factor[i] * (1 + config.padding))
            
            #import copy
            #window_sz = copy.deepcopy(target_sz)
            #window_sz[0] *= (config.scale_factor[x_scale_index] * (1 + config.padding))
            #window_sz[1] *= (config.scale_factor[y_scale_index] * (1 + config.padding))
            #bbox = cxy_wh_2_bbox(target_pos, window_sz)
            
            #print config.scale_factor[x_scale_index], config.scale_factor[y_scale_index], bbox
            #patch_crop[i, :] = crop_chw(im, bbox, config.crop_sz)
        
        patch_crop[0, :] = resize_with_pad_to_square(im, scale_factor_pair, config.crop_sz)

        search = patch_crop - config.net_average_image
        response = net(torch.Tensor(search).cuda())

        np_response = response.cpu().detach().numpy()
        uint16_response = rearrangeMolecules(normalize_to_uint16(np_response[0]).transpose(1,2,0))
        cropped_size = (int(scale_factor_pair[0] * im.shape[0]), int(scale_factor_pair[1] * im.shape[1]))
        cropped_uint16_response = uint16_response[:cropped_size[0], :cropped_size[1], :]

        unwarped_uint16_response = reverse_resize(cropped_uint16_response, scale_factor_pair, (im.shape[1], im.shape[0]))
        
        reponse_heatmap = to_heatmap(rearrangeMolecules(normalize_to_255(np_response[0]).transpose(1,2,0)))
        cropped_size = (int(scale_factor_pair[0] * im.shape[0]), int(scale_factor_pair[1] * im.shape[1]))
        cropped_heatmap = reponse_heatmap[:cropped_size[0], :cropped_size[1], :]
        unwarped_heatmap = reverse_resize(cropped_heatmap, scale_factor_pair, (im.shape[1], im.shape[0]))
        
        cv2.imwrite(os.path.join(output_folder_data_dir, "{:04}.png".format(f + 1)), unwarped_uint16_response)
        cv2.imwrite(os.path.join(output_folder_heatmap_dir, "{:04}.png".format(f + 1)), unwarped_heatmap)

        if visualization:
      #      im0 = rearrangeMolecules(im0)
            transposed_searches = np.array([s.transpose(1,2,0) / 255.0 for s in patch_crop])
            two_d_searches = np.vstack([np.hstack(transposed_searches[i*config.axis_num_scale:(i+1)*config.axis_num_scale, :])
                                                                      for i in range(config.axis_num_scale)])
            
            
            im0 = rearrangeMolecules(np_response[0].transpose(1,2,0)) * 5
            lowest_min = lowest_min if lowest_min < np.min(np_response) else np.min(np_response)
            highest_max = highest_max if highest_max > np.max(np_response) else np.max(np_response)
            normalized_im0 = normalize_to_255(im0)
            #im0_heat = cv2.applyColorMap(((im0 - np.min(im0)) / (np.max(im0) - np.min(im0)) * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
            #cv2.imshow("heat", im0_heat)
            #assert(abs(np.max(im0)) + abs(np.min(im0)) < 1.0)
            rearranged_responses = np.array([to_heatmap(rearrangeMolecules(r.transpose(1,2,0)))
                                             for r in normalize_to_255(np_response)])

            two_d_responses = np.vstack([np.hstack(rearranged_responses[i*config.axis_num_scale:(i+1)*config.axis_num_scale, :])
                                                                        for i in range(config.axis_num_scale)])

            rearranged_responses2 = np.array([to_heatmap(r.transpose(1,2,0))
                                             for r in normalize_to_255(np_response)])

            two_d_responses2 = np.vstack([np.hstack(rearranged_responses2[i*config.axis_num_scale:(i+1)*config.axis_num_scale, :])
                                                                            for i in range(config.axis_num_scale)])

            #print "np_response.shape:", np_response.shape
            
            #input()
            reponse_heatmap = to_heatmap(rearrangeMolecules(normalize_to_255(np_response[0]).transpose(1,2,0)))
            cropped_size = (int(scale_factor_pair[0] * im.shape[0]), int(scale_factor_pair[1] * im.shape[1]))
            #print "reponse_heatmap.shape:", reponse_heatmap.shape
            cropped_heatmap = reponse_heatmap[:cropped_size[0], :cropped_size[1], :]
            unwarped_heatmap = reverse_resize(cropped_heatmap, scale_factor_pair, (im.shape[1], im.shape[0]))
            cv2.imshow("cropped", cropped_heatmap)
            cv2.imshow("unwarped", unwarped_heatmap)

            #np_response = response.cpu().detach().numpy()
            #rearranged_responses = np.array([rearrangeMolecules(r.transpose(1,2,0)) * 5 for r in np_response])
            #reshaped_responses = transposed_searches.reshape((config.num_scale, config.num_scale, -1))
            #all_searches = np.hstack(transposed_searches)
            #all_responses = np.hstack(rearranged_responses)
            #to_display = np.vstack([np.hstack(transposed_searches), np.hstack(rearranged_responses)])
            cv2.imshow("i", two_d_searches)
            cv2.imshow("o2", two_d_responses2)
            cv2.imshow("o", two_d_responses)
            #for index,scaleValue in enumerate(config.scale_factor):
            #    cv2.imshow("i {} {:.2f}".format(index,scaleValue), search[index].transpose(1,2,0) * 5)
            #    imlocal = my_response[index].cpu().detach().numpy().transpose(1,2,0)
            #    imlocal = rearrangeMolecules(imlocal)
            #    cv2.imshow("o {} {:.2f}".format(index,scaleValue), imlocal * 5)
            cv2.waitKey(50)
            #input()

        ### hacks
        peak, idx = torch.max(response.view(config.num_scale, -1), 1)
        peak = peak.data.cpu().numpy() * config.scale_penalties
        best_scale = np.argmax(peak)
        #r_max, c_max = np.unravel_index(idx.cpu()[best_scale], config.net_input_size)

        #if r_max > config.net_input_size[0] / 2:
        #    r_max = r_max - config.net_input_size[0]
        #if c_max > config.net_input_size[1] / 2:
        #    c_max = c_max - config.net_input_size[1]

        #print window_sz.shape
        #input()
        #window_sz = target_sz * (config.scale_factor[best_scale] * (1 + config.padding))
        #window_sz = np.multiply(target_sz, (config.scale_factor_pairs[best_scale] * (1 + config.padding)))

        #target_pos = target_pos + np.array([c_max, r_max]) * window_sz / config.net_input_size
        target_sz = np.minimum(np.maximum(window_sz / (1 + config.padding), min_sz), max_sz)

        # model update
        #window_sz = target_sz * (1 + config.padding)
        #bbox = cxy_wh_2_bbox(target_pos, window_sz)
        #patch = crop_chw(im, bbox, config.crop_sz)
        #target = patch - config.net_average_image
        #net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=config.interp_factor)

        res.append(cxy_wh_2_rect1(target_pos, target_sz))  # 1-index

        if visualization:
            im_show = im  #cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.rectangle(im_show, (int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)),
                          (int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)),
                          (0, 255, 0), 3)
            cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.CV_AA)
            cv2.imshow("video", im_show)
            cv2.waitKey(10)

    toc = time.time() - tic
    fps = n_images / toc
    speed.append(fps)
    video_id = 0
    video = "temp"
    print('{:3d} Video: {:12s} Time: {:3.1f}s\tSpeed: {:3.1f}fps'.format(video_id, video, toc, fps))

    # save result
    #test_path = join('result', dataset, 'DCFNet_test')
    #if not isdir(test_path): makedirs(test_path)
    #result_path = join(test_path, video + '.txt')
    #with open(result_path, 'w') as f:
    #    for x in res:
    #        f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')

    print('***Total Mean Speed: {:3.1f} (FPS)***'.format(np.mean(speed)))

#eval_auc(dataset, 'DCFNet_test', 0, 1)

    # save result
    #test_path = join('result', dataset, 'DCFNet_test')
    #if not isdir(test_path): makedirs(test_path)
    #result_path = join(test_path, video + '.txt')
    #with open(result_path, 'w') as f:
    #    for x in res:
    #        f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')
    return SUCCESS_RETURN_VALUE


def generate_heatmaps_for_video(input_video_folder, bb_hw_pairs, output_folder):

    print "Generating heatmaps: {}, {} -> {}".format(input_video_folder, bb_hw_pairs, output_folder)

    # Read the ground truth bounding boxes for each image in this video
    with open(os.path.join(input_video_folder, "groundtruth_rect.txt"), 'r') as groundtruth_file:
        all_lines = groundtruth_file.readlines()

    # Make sure sure all_lines is defined and is not empty
    if not all_lines:
        raise RuntimeError("{} is empty".format(os.path.join(input_video_folder, "groundtruth_rect.txt")))

    num_images = len(all_lines)
    groundtruth_bbs = []
    for groundtruth_line in all_lines:
        list_of_bb_size_strings = groundtruth_line.strip().replace(' ', '').split(',')
        assert(len(list_of_bb_size_strings) == 4)
        bb_sizes = [float(size_str) for size_str in list_of_bb_size_strings]
        groundtruth_bbs.append(bb_sizes)

    for target_bb_h, target_bb_w in bb_hw_pairs:
        scale_factor_output_dir = os.path.join(output_folder, "bb_h={}_w={}".format(target_bb_h, target_bb_w))
        for target_image_index in range(num_images):
            target_image_output_dir = os.path.join(scale_factor_output_dir, "target_image_{}".format(target_image_index))
            
            gt_bb = groundtruth_bbs[target_image_index]


            print "\nGround truth bounding box for target image index {}:".format(target_image_index), gt_bb
            print "Target bounding box size: h={}, w={}".format(target_bb_h, target_bb_w)

            gt_bb_w = gt_bb[2]
            gt_bb_h = gt_bb[3]
            scale_factor_y =  float(gt_bb_h) / float(target_bb_h)
            scale_factor_x =  float(gt_bb_w) / float(target_bb_w)

            result = generate_heatmap_for_specific_target_and_scale(input_video_folder=input_video_folder,
                                                                    num_images=num_images,
                                                                    scale_factor_pair=(scale_factor_y, scale_factor_x),
                                                                    target_image_index=target_image_index,
                                                                    target_groundtruth_bb=gt_bb,
                                                                    output_folder=target_image_output_dir)
    
            if result == SKIPPED_RETURN_VALUE:
                print "Bounding box height={} and width={} skipped for target image index {}".format(target_bb_h,
                                                                                                     target_bb_w,
                                                                                                     target_image_index)

if __name__ == '__main__':
    # base dataset path and setting
    parser = argparse.ArgumentParser(description='Test DCFNet on OTB')
    parser.add_argument('--dataset', metavar='SET', default='OTB2013',
                        choices=['OTB2013', 'OTB2015'], help='tune on which dataset')
    parser.add_argument('--dataset_folder', metavar='FOLDER', required=True,
                        help='Path of folder containing all dataset video folders')
    parser.add_argument('--output_folder', metavar='FOLDER', required=True,
                        help='Path of folder to write all generated heatmaps for videos in dataset_folder')
    parser.add_argument('--model', metavar='PATH', default='param.pth')
    args = parser.parse_args()

    #dataset = args.dataset
    ##base_path = join('dataset', dataset)
    #json_path = join('dataset', dataset + '.json')
    #annos = json.load(open(json_path, 'r'))
    #videos = sorted(annos.keys())

    import multiprocessing
    torch.set_num_threads(multiprocessing.cpu_count() - 1)

    for max_bb_side_size in [150, 200, 275]:
        min_bb_side_size = 20
        #max_bb_side_size = 150 #275
        bb_side_size_step = 5

        num_bb_side_sizes = int((max_bb_side_size - min_bb_side_size) / bb_side_size_step) + 1
        bb_side_sizes = (np.arange(num_bb_side_sizes) * bb_side_size_step) + min_bb_side_size
        bb_hw_pairs = [np.array((bb_side_sizes[i / num_bb_side_sizes], bb_side_sizes[i % num_bb_side_sizes]))
                              for i in range(num_bb_side_sizes ** 2)]

        abs_dataset_folder = os.path.realpath(args.dataset_folder)
        abs_output_folder = os.path.realpath(args.output_folder)
        print "abs_dataset_folder:", abs_dataset_folder
        for dir_entry in os.listdir(abs_dataset_folder):
            input_video_folder = os.path.join(abs_dataset_folder, dir_entry)
            print "dir_entry:", dir_entry
            if not os.path.isdir(input_video_folder):
                print "not directory:", dir_entry
                continue

            output_dir_for_this_video = os.path.join(abs_output_folder, dir_entry)
            generate_heatmaps_for_video(input_video_folder=input_video_folder,
                                        bb_hw_pairs=bb_hw_pairs,
                                        output_folder=output_dir_for_this_video)

        #(1, 1), 

    sys.exit(0)

    videos = ["2014_08_27_2257_parkedMovingPerson203BedSunnyCloudyWeeds_GrayShirt_Row_CloseCross_PrimaryLog"]
    annos = {videos[0]: {}}
    annos[videos[0]]['image_files'] = [
      #"0001.png", 
      #"0002.png", 
      #"0003.png", 
      #"0004.png", 
      #"0005.png", 
      #"0006.png", 
      #"0007.png", 
      #"0008.png", 
      #"0009.png", 
      #"0010.png", 
      #"0011.png", 
      #"0012.png", 
      "0013.png", 
      "0014.png", 
      "0015.png", 
      "0016.png", 
      "0017.png", 
      "0018.png", 
      "0019.png", 
      "0020.png", 
      "0021.png", 
      "0022.png", 
      "0023.png", 
      "0024.png", 
      "0025.png", 
      "0026.png", 
      "0027.png", 
      "0028.png", 
      "0029.png", 
      "0030.png", 
      "0031.png", 
      "0032.png" ]
    annos[videos[0]]['name'] = videos[0]
    #annos[videos[0]]['init_rect'] = [68.219,227.96,93.729,61.307]
    ##98.789 210.68 82.895 97.544
    #113 171.38 96.316 201.4
    annos[videos[0]]['init_rect'] = [175.44,157.08, 114.4, 215]

    use_gpu = True
    visualization = True

    lowest_min = 100
    highest_max = -100

    # default parameter and load feature extractor network
    config = TrackerConfig()
    net = DCFNet(config)
    net.load_param(args.model)
    net.eval().cuda()

    while 1:
        speed = []
        # loop videos
        for video_id, video in enumerate(videos):  # run without resetting
            video_path_name = annos[video]['name']
            init_rect = np.array(annos[video]['init_rect']).astype(np.float)
            image_files = [join(base_path, video_path_name, 'img', im_f) for im_f in annos[video]['image_files']]
            n_images = len(image_files)

            tic = time.time()  # time start

            target_pos, target_sz = rect1_2_cxy_wh(init_rect)  # OTB label is 1-indexed

            im = cv2.imread(image_files[0])  # HxWxC
            #print image_files[0]
            #input()
            # confine results
            min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
            max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

            # crop template
            print im.shape
            #input()
            window_sz = target_sz * (1 + config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)
            patch = crop_chw(im, bbox, (300, 300))
            patch = pad_to_size_centered(patch, config.crop_sz)
            #patch = resize_with_pad_to_square(im, config.crop_sz)
            #transposed_patch = np.array([s.transpose(1,2,0) / 255.0 for s in patch_crop])
            if visualization:
                cv2.imshow("patch", (patch.transpose(1,2,0) / 255.0))
                cv2.waitKey(500)
                print patch.shape

            target = patch - config.net_average_image
            net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())

            #print config.scale_penalties
            #np.set_printoptions(linewidth=120)
            #print np.array(config.scale_penalties).reshape((9,9))
            #input()

            res = [cxy_wh_2_rect1(target_pos, target_sz)]  # save in .txt
            patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)
            for f in range(1, n_images):  # track
                im = cv2.imread(image_files[f])
                #print config.scale_factor

                #for i in range(config.num_scale):  # crop multi-scale search region
                    #x_scale_index = i / config.axis_num_scale  # Integer division
                    #y_scale_index = i % config.axis_num_scale

                    #window_sz = target_sz * (config.scale_factor[i] * (1 + config.padding))
                    
                    #import copy
                    #window_sz = copy.deepcopy(target_sz)
                    #window_sz[0] *= (config.scale_factor[x_scale_index] * (1 + config.padding))
                    #window_sz[1] *= (config.scale_factor[y_scale_index] * (1 + config.padding))
                    #bbox = cxy_wh_2_bbox(target_pos, window_sz)
                    
                    #print config.scale_factor[x_scale_index], config.scale_factor[y_scale_index], bbox
                    #patch_crop[i, :] = crop_chw(im, bbox, config.crop_sz)
                
                patch_crop[0, :] = resize_with_pad_to_square(im, config.scale_factors, config.crop_sz)

                search = patch_crop - config.net_average_image
                response = net(torch.Tensor(search).cuda())

                if visualization:
                    

                    def rearrangeMolecules(im):
                        im_h = im.shape[0]
                        im_w = im.shape[1]
                        return np.vstack([np.hstack([im[im_h/2:im_h,im_w/2:im_w,:],im[im_h/2:im_h,0:im_w/2,:]]),np.hstack([im[0:im_h/2,im_w/2:im_w,:],im[0:im_h/2,0:im_w/2,:]])])

                    def normalize_to_255(im):
                        return (((im * 2.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)

                    def to_heatmap(im):
                        return apply_matplotlib_colormap(im )

              #      im0 = rearrangeMolecules(im0)
                    transposed_searches = np.array([s.transpose(1,2,0) / 255.0 for s in patch_crop])
                    two_d_searches = np.vstack([np.hstack(transposed_searches[i*config.axis_num_scale:(i+1)*config.axis_num_scale, :])
                                                                              for i in range(config.axis_num_scale)])
                    
                    np_response = response.cpu().detach().numpy()
                    im0 = rearrangeMolecules(np_response[0].transpose(1,2,0)) * 5
                    lowest_min = lowest_min if lowest_min < np.min(np_response) else np.min(np_response)
                    highest_max = highest_max if highest_max > np.max(np_response) else np.max(np_response)
                    print "lowest_min", lowest_min
                    print "highest_max", highest_max
                    normalized_im0 = normalize_to_255(im0)
                    print "normalized min", np.min(normalized_im0)
                    print "normalized max", np.max(normalized_im0)
                    #im0_heat = cv2.applyColorMap(((im0 - np.min(im0)) / (np.max(im0) - np.min(im0)) * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
                    #cv2.imshow("heat", im0_heat)
                    #assert(abs(np.max(im0)) + abs(np.min(im0)) < 1.0)
                    rearranged_responses = np.array([to_heatmap(rearrangeMolecules(r.transpose(1,2,0)))
                                                     for r in normalize_to_255(np_response)])

                    two_d_responses = np.vstack([np.hstack(rearranged_responses[i*config.axis_num_scale:(i+1)*config.axis_num_scale, :])
                                                                                for i in range(config.axis_num_scale)])

                    rearranged_responses2 = np.array([to_heatmap(r.transpose(1,2,0))
                                                     for r in normalize_to_255(np_response)])

                    two_d_responses2 = np.vstack([np.hstack(rearranged_responses2[i*config.axis_num_scale:(i+1)*config.axis_num_scale, :])
                                                                                    for i in range(config.axis_num_scale)])

                    print "np_response.shape:", np_response.shape
                    
                    #input()
                    reponse_heatmap = to_heatmap(rearrangeMolecules(normalize_to_255(np_response[0]).transpose(1,2,0)))
                    cropped_size = (int(config.scale_factors[0] * im.shape[0]), int(config.scale_factors[1] * im.shape[1]))
                    print "reponse_heatmap.shape:", reponse_heatmap.shape
                    cropped_heatmap = reponse_heatmap[:cropped_size[0], :cropped_size[1], :]
                    unwarped_heatmap = reverse_resize(cropped_heatmap, config.scale_factors, (720,480))
                    cv2.imshow("cropped", cropped_heatmap)
                    cv2.imshow("unwarped", unwarped_heatmap)

                    #np_response = response.cpu().detach().numpy()
                    #rearranged_responses = np.array([rearrangeMolecules(r.transpose(1,2,0)) * 5 for r in np_response])
                    #reshaped_responses = transposed_searches.reshape((config.num_scale, config.num_scale, -1))
                    #all_searches = np.hstack(transposed_searches)
                    #all_responses = np.hstack(rearranged_responses)
                    #to_display = np.vstack([np.hstack(transposed_searches), np.hstack(rearranged_responses)])
                    cv2.imshow("i", two_d_searches)
                    cv2.imshow("o2", two_d_responses2)
                    print "two_d_responses2.shape:", two_d_responses2.shape 
                    cv2.imshow("o", two_d_responses)
                    #for index,scaleValue in enumerate(config.scale_factor):
                    #    cv2.imshow("i {} {:.2f}".format(index,scaleValue), search[index].transpose(1,2,0) * 5)
                    #    imlocal = my_response[index].cpu().detach().numpy().transpose(1,2,0)
                    #    imlocal = rearrangeMolecules(imlocal)
                    #    cv2.imshow("o {} {:.2f}".format(index,scaleValue), imlocal * 5)
                    cv2.waitKey(500)
                    #input()

                ### hacks
                peak, idx = torch.max(response.view(config.num_scale, -1), 1)
                peak = peak.data.cpu().numpy() * config.scale_penalties
                best_scale = np.argmax(peak)
                #r_max, c_max = np.unravel_index(idx.cpu()[best_scale], config.net_input_size)

                #if r_max > config.net_input_size[0] / 2:
                #    r_max = r_max - config.net_input_size[0]
                #if c_max > config.net_input_size[1] / 2:
                #    c_max = c_max - config.net_input_size[1]

                #print window_sz.shape
                #input()
                #window_sz = target_sz * (config.scale_factor[best_scale] * (1 + config.padding))
                #window_sz = np.multiply(target_sz, (config.scale_factor_pairs[best_scale] * (1 + config.padding)))

                #target_pos = target_pos + np.array([c_max, r_max]) * window_sz / config.net_input_size
                target_sz = np.minimum(np.maximum(window_sz / (1 + config.padding), min_sz), max_sz)

                # model update
                #window_sz = target_sz * (1 + config.padding)
                #bbox = cxy_wh_2_bbox(target_pos, window_sz)
                #patch = crop_chw(im, bbox, config.crop_sz)
                #target = patch - config.net_average_image
                #net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=config.interp_factor)

                res.append(cxy_wh_2_rect1(target_pos, target_sz))  # 1-index

                if visualization:
                    im_show = im  #cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(im_show, (int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)),
                                  (int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)),
                                  (0, 255, 0), 3)
                    cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.CV_AA)
                    cv2.imshow("video", im_show)
                    cv2.waitKey(1000)

            toc = time.time() - tic
            fps = n_images / toc
            speed.append(fps)
            print('{:3d} Video: {:12s} Time: {:3.1f}s\tSpeed: {:3.1f}fps'.format(video_id, video, toc, fps))

            # save result
            #test_path = join('result', dataset, 'DCFNet_test')
            #if not isdir(test_path): makedirs(test_path)
            #result_path = join(test_path, video + '.txt')
            #with open(result_path, 'w') as f:
            #    for x in res:
            #        f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')

        print('***Total Mean Speed: {:3.1f} (FPS)***'.format(np.mean(speed)))

        #eval_auc(dataset, 'DCFNet_test', 0, 1)
