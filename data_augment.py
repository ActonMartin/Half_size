import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

from PIL import Image
from functools import reduce
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def sometimes(aug): return iaa.Sometimes(0.9, aug)


def sometimes_p1(aug): return iaa.Sometimes(0.9, aug)


def load_image(data_name, is_heatmap, input_size):

    # dummy function, implement this
    # Return a numpy array of shape (N, height, width, #channels)
    # or a list of (height, width, #channels) arrays (may have different image
    # sizes).
    # Images should be in RGB for colorspace augmentations.
    # (cv2.imread() returns BGR!)
    # Images should usually be in uint8 with values from 0-255.
    # return np.zeros((128, 32, 32, 3), dtype=np.uint8) + (batch_idx % 255)

    im1 = Image.open(data_name).convert('L')
    im1 = np.array(im1)
    '''
    if im1.shape[0] > im1.shape[1]:
        # 原型：cv2.flip(src, flipCode[, dst]) → dst  flipCode表示对称轴 0：x轴  1：y轴.
        # -1：both
        im1 = cv2.flip(im1, 1)
        im1 = cv2.transpose(im1)
    # resize(img, (width, height))
    '''
    im1 = cv2.resize(im1, (input_size[1], input_size[0]), cv2.INTER_NEAREST)

    if not is_heatmap:
        im1 = im1[np.newaxis, :, :, np.newaxis]
    else:
        im1 = im1[np.newaxis, :, :, np.newaxis].astype(np.float32) / 255
    return im1


'''
def train_on_images(images,idx,i):

    # dummy function, implement this

    #images=cv2.cvtColor(images, cv2.COLOR_RGB2BGR)

    cv2.imwrite('./images/'+str(idx)+"_"+str(i)+'_messigray.jpg',images)

    pass
'''


def get_seq():
    seq = iaa.Sequential([
        # iaa.Fliplr(0.5),  # horizontal flips
        # iaa.Flipud(0.5), # horizontal flips
        # iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        # iaa.Sometimes(0.5,
        #    iaa.GaussianBlur(sigma=(0, 0.2))
        # ),
        # Strengthen or weaken the contrast in each image.
        # iaa.ContrastNormalization((0.9, 1.1)),##原先是 0.8,1.2
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        #iaa.AdditiveGaussianNoise(loc=0, scale=(0,0.02*255), per_channel=0.2),
        sometimes_p1(iaa.SaltAndPepper(p=0.00, per_channel=0.5)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        #iaa.Multiply((0.95, 1.05), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        sometimes(iaa.Affine(
            # scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to
            # 80-120% of their size, individually per axis ##原先是关闭的
            # translate by -20 to +20 percent (per axis)
            translate_percent={"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
            # rotate=(-3, 3), # rotate by -45 to +45 degrees
            # shear=(-3, 3), # shear by -16 to +16 degrees
            # use nearest neighbour or bilinear interpolation (fast)
            order=[0, 1],
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd
            # image from the top for examples)
        )), ], random_order=True)  # apply augmenters in random order
    return seq


def aug_data(seq, image, heatmap):
    # cv2.imwrite('./images/'+'messigray.jpg',images)
    # seq_det = seq.to_deterministic() # call this for each batch again, NOT
    # only once at the start
    image_aug, heatmap = seq(
        images=image, heatmaps=heatmap)  # done by the library
    ##train_on_images(images_aug,batch_idx, 0)
    return image_aug, heatmap
