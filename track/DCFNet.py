from os.path import join, isdir
from os import makedirs
import argparse
import json
import numpy as np
import torch

import cv2
import matplotlib.pyplot as plt
import time as time
from util import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox
from net import DCFNet
from eval_otb import eval_auc

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
    crop_sz = 1440

    lambda0 = 1e-4
    padding = 2
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

    net_input_size = [crop_sz, crop_sz]
    net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, net_input_size)
    yf = torch.rfft(torch.Tensor(y).view(1, 1, crop_sz, crop_sz).cuda(), signal_ndim=2)
    cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()


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


if __name__ == '__main__':
    # base dataset path and setting
    parser = argparse.ArgumentParser(description='Test DCFNet on OTB')
    parser.add_argument('--dataset', metavar='SET', default='OTB2013',
                        choices=['OTB2013', 'OTB2015'], help='tune on which dataset')
    parser.add_argument('--model', metavar='PATH', default='param.pth')
    args = parser.parse_args()

    dataset = args.dataset
    base_path = join('dataset', dataset)
    json_path = join('dataset', dataset + '.json')
    annos = json.load(open(json_path, 'r'))
    videos = sorted(annos.keys())


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
    #annos[videos[0]]['init_rect'] = [68, 228, 94, 61]
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
        window_sz = target_sz * (1 + config.padding)
        bbox = cxy_wh_2_bbox(target_pos, window_sz)
        patch = crop_chw(im, bbox, config.crop_sz)
        print patch.shape
        input

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

            for i in range(config.num_scale):  # crop multi-scale search region
                x_scale_index = i / config.axis_num_scale  # Integer division
                y_scale_index = i % config.axis_num_scale

                #window_sz = target_sz * (config.scale_factor[i] * (1 + config.padding))
                import copy
                window_sz = copy.deepcopy(target_sz)
                window_sz[0] *= (config.scale_factor[x_scale_index] * (1 + config.padding))
                window_sz[1] *= (config.scale_factor[y_scale_index] * (1 + config.padding))
                bbox = cxy_wh_2_bbox(target_pos, window_sz)
                #print config.scale_factor[x_scale_index], config.scale_factor[y_scale_index], bbox
                patch_crop[i, :] = crop_chw(im, bbox, config.crop_sz)

            search = patch_crop - config.net_average_image
            response = net(torch.Tensor(search).cuda())
            ## hacks
            #my_response = response
            #my_response = response.view(config.num_scale, -1) 
            #print response.shape
            #print search.shape
        #    print patch_crop.shape
            #print my_response[0]
            #patch_crop_im0 = patch_crop[0].transpose(1,2,0)
            #print patch_crop_im0
       #     im0 = my_response[0].cpu().detach().numpy().transpose(1,2,0)
            def rearrangeMolecules(im):
                im_h = im.shape[0]
                im_w = im.shape[1]
                return np.vstack([np.hstack([im[im_h/2:im_h,im_w/2:im_w,:],im[im_h/2:im_h,0:im_w/2,:]]),np.hstack([im[0:im_h/2,im_w/2:im_w,:],im[0:im_h/2,0:im_w/2,:]])])

            def normalize_to_255(im):
                return ((im + 1.0) / 2.0 * 255.0).astype(np.uint8)

            def to_heatmap(im):
                return apply_matplotlib_colormap(im, )

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

            #np_response = response.cpu().detach().numpy()
            #rearranged_responses = np.array([rearrangeMolecules(r.transpose(1,2,0)) * 5 for r in np_response])
            #reshaped_responses = transposed_searches.reshape((config.num_scale, config.num_scale, -1))
            #all_searches = np.hstack(transposed_searches)
            #all_responses = np.hstack(rearranged_responses)
            #to_display = np.vstack([np.hstack(transposed_searches), np.hstack(rearranged_responses)])
            cv2.imshow("i", two_d_searches)
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
            window_sz = target_sz * (1 + config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)
            patch = crop_chw(im, bbox, config.crop_sz)
            target = patch - config.net_average_image
            net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=config.interp_factor)

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
        test_path = join('result', dataset, 'DCFNet_test')
        if not isdir(test_path): makedirs(test_path)
        result_path = join(test_path, video + '.txt')
        with open(result_path, 'w') as f:
            for x in res:
                f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')

    print('***Total Mean Speed: {:3.1f} (FPS)***'.format(np.mean(speed)))

    eval_auc(dataset, 'DCFNet_test', 0, 1)
