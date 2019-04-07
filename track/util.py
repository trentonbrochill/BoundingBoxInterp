import numpy as np
import cv2


def cxy_wh_2_rect1(pos, sz):
    return np.array([pos[0]-sz[0]/2+1, pos[1]-sz[1]/2+1, sz[0], sz[1]])  # 1-index


def rect1_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2-1, rect[1]+rect[3]/2-1]), np.array([rect[2], rect[3]])  # 0-index


def cxy_wh_2_bbox(cxy, wh):
    return np.array([cxy[0]-wh[0]/2, cxy[1]-wh[1]/2, cxy[0]+wh[0]/2, cxy[1]+wh[1]/2])  # 0-index


def cxy_wh_2_bbox_w_h_separate(cxy, w, h):
    '''Center (x,y), width, and height to bounding box.
    
    Returns:
        np.array of size 4 where index 0 is x coordinate of left edge of BB, index 1 is y coordinate of top edge of BB,
        index 2 is x coordinate of right edge of BB, and index 3 is y coordinate of bottom edge of BB
    '''
    return np.array([cxy[0]-wh[0]/2, cxy[1]-wh[1]/2, cxy[0]+wh[0]/2, cxy[1]+wh[1]/2])  # 0-index


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g


def crop_chw(image, bbox, out_sz, padding=(0,0,0)):
    a = (out_sz[1] - 1) / (bbox[2]-bbox[0])
    b = (out_sz[0] - 1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, out_sz, borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1))


def resize_with_pad_to_square(image, scale_factors, out_sz, padding=(0,0,0)):
    # (bbox[2] - bbox[0]) is the width of the bbox, so |a| is the factor by which the output width is greater than 
    # (or less than) the bbox width.
    a = (float(out_sz[1])-1) / (image.shape[1])
    # (bbox[3] - bbox[1]) is the height of the bbox, so |b| is the factor by which the output height is greater than 
    # (or less than) the bbox height.
    b = (float(out_sz[0])-1) / (image.shape[0])

    assert(scale_factors[0] * image.shape[0] <= out_sz[1])
    assert(scale_factors[1] * image.shape[1] <= out_sz[0])

    scale = min(a, b)

    #print "a:", a, ", b:", b
    #input()

    # 
    #c = -a * bbox[0]
    c = 0
    #d = -b * bbox[1]
    d = 0
    mapping = np.array([[scale_factors[1],                0, c],
                        [               0, scale_factors[0], d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, out_sz, borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1))


def reverse_resize(image, scale_factors, out_sz, padding=(0,0,0)):
    # (bbox[2] - bbox[0]) is the width of the bbox, so |a| is the factor by which the output width is greater than 
    # (or less than) the bbox width.
    #a = (float(out_sz[1])-1) / (image.shape[1])
    # (bbox[3] - bbox[1]) is the height of the bbox, so |b| is the factor by which the output height is greater than 
    # (or less than) the bbox height.
    #b = (float(out_sz[0])-1) / (image.shape[0])

    assert((1.0 / scale_factors[0]) * image.shape[0] <= out_sz[1])
    assert((1.0 / scale_factors[1]) * image.shape[1] <= out_sz[0])

    #scale = min(a, b)

    #print "a:", a, ", b:", b
    #input()

    # 
    #c = -a * bbox[0]
    c = 0
    #d = -b * bbox[1]
    d = 0
    mapping = np.array([[1.0 / scale_factors[1],                      0, c],
                        [                     0, 1.0 / scale_factors[0], d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, out_sz, borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def resize_with_pad_to_square_centered(image, scale_factors, out_sz, padding=(0,0,0)):
    # (bbox[2] - bbox[0]) is the width of the bbox, so |a| is the factor by which the output width is greater than 
    # (or less than) the bbox width.
    a = (float(out_sz[1])-1) / (image.shape[1])
    # (bbox[3] - bbox[1]) is the height of the bbox, so |b| is the factor by which the output height is greater than 
    # (or less than) the bbox height.
    b = (float(out_sz[0])-1) / (image.shape[0])

    assert(scale_factors[0] * image.shape[0] <= out_sz[1])
    assert(scale_factors[1] * image.shape[1] <= out_sz[0])

    scale = min(a, b)

    #print "a:", a, ", b:", b
    #input()

    # 
    #c = -a * bbox[0]
    #c = 0

    print "scale:", out_sz
    print "out_sz:", out_sz
    print "image.shape:", image.shape


    c = (out_sz[0] / 2) - ((image.shape[1] * scale_factors[1])  / 2)
    #d = -b * bbox[1]
    #d = 0
    d = (out_sz[1] / 2) - ((image.shape[0] * scale_factors[0])  / 2)

    mapping = np.array([[scale_factors[1],                0, c],
                        [               0, scale_factors[0], d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, out_sz, borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1))


def pad_to_size_centered(image, out_sz, padding=(0,0,0)):

    img_t = np.transpose(image, (1, 2, 0))

    print img_t.shape

    #c = -a * bbox[0]
    c = (out_sz[0] / 2) - (img_t.shape[1]  / 2)
    #d = -b * bbox[1]
    d = (out_sz[1] / 2) - (img_t.shape[0]  / 2)

    print "c:", c, ", d:", d

    mapping = np.array([[1, 0, c],
                        [0, 1, d]]).astype(np.float)
    crop = cv2.warpAffine(img_t, mapping, out_sz, borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1))



if __name__ == '__main__':
    a = gaussian_shaped_labels(10, [5,5])
#    print a
