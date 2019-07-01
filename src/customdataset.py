import pdb
import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from skimage.feature import canny
from skimage.color import rgb2gray


class Dataset(object):
    def __init__(self, config):
        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS

    def get_item(self, img, mask):
        """ TODO: this is the slow version of code. Since canny operation is from skimage, now is tensor -> img -> tensor
        :param img: (bs, 3, h, w)
        :param mask: (bs, 3, h, w)
        :return: img(bs, 3, h, w), img_gray(bs, 1, h, w), edge(bs, 1, h, w), mask(bs, 1, h, w)
        """
        bs, _, h, w = img.shape
        imgs, masks, imgs_gray, edges = [], [], [], []
        for bs_i in range(bs):
            img_i = self.tensor2img(img[bs_i])
            mask_i = self.load_mask(self.tensor2img(mask[bs_i]))
            img_gray_i = rgb2gray(img_i)
            edge_i = self.load_edge(img_gray_i, mask_i)
            imgs.append(self.to_tensor(img_i))
            imgs_gray.append(self.to_tensor(img_gray_i))
            edges.append(self.to_tensor(edge_i))
            masks.append(self.to_tensor(mask_i))

        return torch.stack(imgs), torch.stack(imgs_gray), torch.stack(edges), torch.stack(masks)

    def load_mask(self, mask):
        mask = rgb2gray(mask)
        mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def load_edge(self, img, mask):
        sigma = self.sigma
        mask = (1 - mask / 255).astype(np.bool)
        return canny(img, sigma=sigma, mask=mask).astype(np.float)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def tensor2img(self, x):
        """
        :param x: (3, h, w)  [0, 1]
        :return: out
        """
        out = (x.permute(1, 2, 0).numpy()*255).astype(np.uint8)
        return out
