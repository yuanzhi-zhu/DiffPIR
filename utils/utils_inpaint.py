# -*- coding: utf-8 -*-
import numpy as np
import torch
from utils import utils_image as util 

'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
'''


# --------------------------------
# get rho and sigma
# --------------------------------
def get_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma2=2.55):
    '''
    Kai Zhang (github: https://github.com/cszn)
    03/03/2019
    '''
    modelSigma1 = 49.0
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num)
    sigmas = modelSigmaS/255.
    mus = list(map(lambda x: (sigma**2)/(x**2)/3, sigmas))
    rhos = mus
    return rhos, sigmas


def shepard_initialize(image, measurement_mask, window=5, p=2):
    wing = np.floor(window/2).astype(int) # Length of each "wing" of the window.
    h, w = image.shape[0:2]
    ch = 3 if image.ndim == 3 and image.shape[-1] == 3 else 1
    x = np.copy(image) # ML initialization
    for i in range(h):
        i_lower_limit = -np.min([wing, i])
        i_upper_limit = np.min([wing, h-i-1])
        for j in range(w):
           if measurement_mask[i, j] == 0: # checking if there's a need to interpolate
               j_lower_limit = -np.min([wing, j])
               j_upper_limit = np.min([wing, w-j-1])

               count = 0 # keeps track of how many measured pixels are withing the neighborhood
               sum_IPD = 0
               interpolated_value = 0

               num_zeros = window**2
               IPD = np.zeros([num_zeros])
               pixel = np.zeros([num_zeros,ch])

               for neighborhood_i in range(i+i_lower_limit, i+i_upper_limit):
                   for neighborhood_j in range(j+j_lower_limit, j+j_upper_limit):
                      if measurement_mask[neighborhood_i, neighborhood_j] == 1:
                          # IPD: "inverse pth-power distance".
                          IPD[count] = 1.0/((neighborhood_i - i)**p + (neighborhood_j - j)**p)
                          sum_IPD = sum_IPD + IPD[count]
                          pixel[count] = image[neighborhood_i, neighborhood_j]
                          count = count + 1

               for c in range(count):
                   weight = IPD[c]/sum_IPD
                   interpolated_value = interpolated_value + weight*pixel[c]
               x[i, j] = interpolated_value

    return x

### Mask generator from https://github.com/DPS2022/diffusion-posterior-sampling/

def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w

class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask


if __name__ == '__main__':
    # image path & sampling ratio
    import matplotlib.pyplot as mplot
    import matplotlib.image as mpimg
    Im = mpimg.imread('test.bmp')
    #Im = Im[:,:,1]
    Im = np.squeeze(Im)

    SmpRatio = 0.2
    # creat mask
    mask_Array = np.random.rand(Im.shape[0],Im.shape[1])
    mask_Array = (mask_Array < SmpRatio)
    print(mask_Array.dtype)

    # sampled image
    print('The sampling ratio is', SmpRatio)
    Im_sampled = np.multiply(np.expand_dims(mask_Array,2), Im)
    util.imshow(Im_sampled)
    
    a = shepard_initialize(Im_sampled.astype(np.float32), mask_Array, window=9)
    a = np.clip(a,0,255)


    print(a.dtype)
    
    
    util.imshow(np.concatenate((a,Im_sampled),1)/255.0)
    util.imsave(np.concatenate((a,Im_sampled),1),'a.png')

    


