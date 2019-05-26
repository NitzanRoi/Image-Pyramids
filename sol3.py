import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import color
from skimage import io
import os


GREYSCALE_CODE = 1
RGB_CODE = 2
MAX_PIXEL_VAL = 255
INVALID_FILTER_SIZE_MSG = "Error: the given size of the filter is not valid"
INVALID_PYRAMID_SIZE_MSG = "Error: the given pyramid list is empty"
INVALID_COEFF_MSG = "Error: the coefficients array is empty"
INVALID_COEFF_SIZE_MSG = "Error: the given size of the coeff list is not valid"
INVALID_LEVELS_MSG = "Error: levels is 0 so there is no image to show"
INVALID_IMAGES_SIZES_MSG = "Error: im1 and im2 and mask don't have the same dimensions"
GAUSSIAN_BASIC_KERNEL = np.array([1, 1])
MIN_DIM_GAUSSIAN = 16


def read_image(filename, representation):
    """
    this function reads an image file and returns it in a given representation
    filename is the image
    representation code: 1 is greyscale, 2 is RGB
    returns an image
    """
    final_img = io.imread(filename).astype(np.float64)
    if (representation == GREYSCALE_CODE):
        final_img = color.rgb2gray(final_img)
    final_img /= MAX_PIXEL_VAL
    return final_img.astype(np.float64)


def is_even_number(num):
    """
    return boolean - if num is even number
    """
    return (num % 2 == 0)


def create_gaussian(kernel_size):
    """
    calculates the gaussian using binomial coefficients and returns it
    """
    gaussian = GAUSSIAN_BASIC_KERNEL.copy()
    for i in range(kernel_size - 2):
        gaussian = signal.convolve(gaussian, GAUSSIAN_BASIC_KERNEL)
    if (np.sum(gaussian) != 0):
        return gaussian / np.sum(gaussian)
    return gaussian


def blur(im, kernel):
    """
    performs image blurring using convolution between image and Gaussian filter
    (once as a row vector and once as a column vector)
        im - the image, float64 greyscale
        kernel - the gaussian kernel
    returns output as float64 greyscale
    """
    row_blur = signal.convolve2d(im, kernel, mode='same').astype(np.float64)
    return signal.convolve2d(row_blur, kernel.T, mode='same').astype(np.float64)


def get_even_idx_values(row):
    """
    given row - returns its values which are in the even indexes
    """
    even_indexes = np.arange(0, row.shape[0], 2)
    return np.take(row, even_indexes)


def down_sample(image):
    """
    returns down-sampled image (i.e. reduce function)
    """
    return np.apply_along_axis(get_even_idx_values, 1, image[::2])


def helper_pad_expand(origin_shape, expand_im):
    """
    helper function to pad the expanded image
    """
    if (is_even_number(origin_shape[1])):
        expand_im = np.pad(expand_im, [(0, 0), (0, 1)], 'constant')
    if (is_even_number(origin_shape[0])):
        expand_im = np.pad(expand_im, [(0, 1), (0, 0)], 'constant')
    return expand_im


def up_sample(image):
    """
    returns up-sampled image (i.e. expand function)
    """
    indices_cols = np.arange(1, image.shape[1])
    indices_rows = np.arange(1, image.shape[0])
    return np.insert(np.insert(image, indices_cols, 0, axis=1), indices_rows, 0, axis=0)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    constructs a Gaussian pyramid with the given image
        im - a grayscale float64 double image, values in range [0, 1]
        max_levels - maximal number of levels in the resulting pyramid
                    (including the given image)
        filter_size - the size of the filter, an odd number
    returns:
        pyr - standard *python* array with max length of max_levels parameter
              each element is a greyscale image
        filter_vec - a row vector of shape (1, filter_size), normalized
    """
    if (filter_size <= 1 or is_even_number(filter_size)):
        print(INVALID_FILTER_SIZE_MSG)
        exit()
    pyr = [im]
    filter_vec = create_gaussian(filter_size).reshape(1, filter_size)
    for i in range(max_levels - 1):
        im = blur(im, filter_vec)
        im = down_sample(im)
        minimum_stop_condition = min(im.shape[0], im.shape[1])
        if (minimum_stop_condition < MIN_DIM_GAUSSIAN):
            break
        pyr.append(im)
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    constructs a Laplacian pyramid with the given image
        im - a grayscale float64 double image, values in range [0, 1]
        max_levels - maximal number of levels in the resulting pyramid
                    (including the given image)
        filter_size - the size of the filter, an odd number
    returns:
        pyr - standard *python* array with max length of max_levels parameter
              each element is a greyscale image
        filter_vec - a row vector of shape (1, filter_size), normalized
    """
    if (filter_size <= 1 or is_even_number(filter_size)):
        print(INVALID_FILTER_SIZE_MSG)
        exit()
    pyr = []
    gaussian_list, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(len(gaussian_list)): # the max-levels and min-levels check is in the gaussian function
        if (i == len(gaussian_list) - 1):
            pyr.append(gaussian_list[i])
        else:
            expand_pyramid = up_sample(gaussian_list[i + 1])
            expand_pyramid = helper_pad_expand(gaussian_list[i].shape, expand_pyramid)
            expand_pyramid = blur(expand_pyramid, (2 * filter_vec))
            pyr.append(gaussian_list[i] - expand_pyramid)
    return pyr, filter_vec


def mult_coeff(lpyr, coeff):
    """
    returns lpyr multiplied with coeff
    """
    for i in range(len(lpyr)):
        lpyr[i] = np.multiply(lpyr[i], coeff[i])
    return lpyr


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    returns the reconstruction of an image from its laplacian pyramid
        lpyr - laplacian pyramid
        filter_vec - the filter of the blur
        coeff - python list, before reconstruct an image we multiply each level i
                from the laplacian pyramid by its corresponding coefficient in coeff[i]
    """
    if (len(lpyr) == 0):
        print(INVALID_PYRAMID_SIZE_MSG)
        exit()
    if (len(coeff) == 0):
        print(INVALID_COEFF_MSG)
        exit()
    if (len(coeff) != len(lpyr)):
        print(INVALID_COEFF_SIZE_MSG)
        exit()
    lpyr = mult_coeff(lpyr, coeff)
    cur_gaussian = lpyr[len(lpyr) - 1]
    for i in range(len(lpyr) - 1):
        expand_laplacian = up_sample(cur_gaussian)
        expand_laplacian = helper_pad_expand(lpyr[len(lpyr) - 2 - i].shape, expand_laplacian)
        expand_laplacian = blur(expand_laplacian, (2 * filter_vec))
        cur_gaussian = lpyr[len(lpyr) - 2 - i] + expand_laplacian
    return cur_gaussian


def render_pyramid(pyr, levels):
    """
    calculates the 'res' black image dimensions and returns it
    """
    res_height = pyr[0].shape[0]
    res_width = 0
    for i in range(min(levels, len(pyr))):
        res_width += pyr[i].shape[1]
    return np.zeros((res_height, res_width))


def stretch_pyramid_values(pyr):
    """
    given pyramid list - stretch all its images to range [0, 1],
    by subtracting the minimal value from every pixel and then dividing every pixel
    by max-min
    """
    for i in range(len(pyr)):
        max_diff = np.max(pyr[i]) - np.min(pyr[i])
        pyr[i] -= np.min(pyr[i])
        if (max_diff != 0):
            pyr[i] /= max_diff
    return pyr


def display_pyramid(pyr, levels):
    """
    displays the given pyramid
    """
    if (len(pyr) == 0):
        print(INVALID_PYRAMID_SIZE_MSG)
        exit()
    if (levels == 0):
        print(INVALID_LEVELS_MSG)
        exit()
    res = render_pyramid(pyr, levels)
    pyr = stretch_pyramid_values(pyr)
    col_index = 0
    for i in range(min(levels, len(pyr))):
        res[0:pyr[i].shape[0], col_index:(col_index+pyr[i].shape[1])] = pyr[i]
        col_index += pyr[i].shape[1]
    plt.imshow(res, cmap=plt.get_cmap('gray'))
    plt.show()


def build_laplacian_for_blend(L_1, L_2, G_m):
    """
    given L_1, L_2, G_m - construct Laplacian pyramid for each level
    """
    L_out = []
    for i in range(len(L_1)):
        L_out.append(np.multiply(G_m[i], L_1[i]) + np.multiply((1 - G_m[i]), L_2[i]))
    return L_out


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    blends 2 input greyscale images (im1 and im2) according to the mask boolean matrix
        max_levels - for the creating of the Gaussian and Laplacian pyramids
        filter_size_im - the size of the Gaussian filter (an odd number) which is used
                         in the construction of the Laplacian of im1 and im2
        filter_size_mask - the size of the Gaussian filter (an odd number) which is used
                           in the construction of the Gaussian of mask
    """
    if (is_even_number(filter_size_im) or is_even_number(filter_size_mask)):
        print(INVALID_FILTER_SIZE_MSG)
        exit()
    if (filter_size_im <= 1 or filter_size_mask <= 1):
        print(INVALID_FILTER_SIZE_MSG)
        exit()
    if (im1.shape != im2.shape or im1.shape != mask.shape or im2.shape != mask.shape):
        print(INVALID_IMAGES_SIZES_MSG)
        exit()
    L_1, filter = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L_2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    G_m = build_gaussian_pyramid(np.array(mask, dtype=np.float64), max_levels, filter_size_mask)[0]
    L_out = build_laplacian_for_blend(L_1, L_2, G_m)
    im_blend = laplacian_to_image(L_out, filter, [1] * len(L_out))
    return np.clip(im_blend, 0, 1)


def relpath(filename):
    """
    helper function to load images
    """
    return os.path.join(os.path.dirname(__file__), filename)


def plot_images(images_arr):
    """
    given array of 4 images (2 input images, mask and blend result)
    plot them in 4 quarters
    """
    plt.figure()
    im1_fig = plt.subplot(2, 2, 1)
    im2_fig = plt.subplot(2, 2, 2)
    mask_fig = plt.subplot(2, 2, 3)
    blend_fig = plt.subplot(2, 2, 4)
    im1_fig.imshow(images_arr[0], cmap=plt.get_cmap('gray'))
    im2_fig.imshow(images_arr[1], cmap=plt.get_cmap('gray'))
    mask_fig.imshow(images_arr[2], cmap=plt.get_cmap('gray'))
    blend_fig.imshow(images_arr[3], cmap=plt.get_cmap('gray'))
    plt.show()


def blending_example1():
    """
    performs pyramid blending and returns:
        im1 and im2 - the images from the blending
        mask - the binary mask
        im_blend - the result
    """
    max_levels = 15
    filter_size_im = 5
    filter_size_mask = 3
    im1 = read_image(relpath("externals/bg1.jpg"), RGB_CODE) # the background img
    im2 = read_image(relpath("externals/obj1.jpg"), RGB_CODE) # the object-masked img
    mask = np.round(read_image(relpath("externals/mask1.jpg"), # the mask
                               GREYSCALE_CODE)).astype(np.bool)
    im_blend = np.zeros((im1.shape))
    for i in range(3):
        im_blend[:,:,i] = pyramid_blending(im1[:,:,i], im2[:,:,i],
                                           mask, max_levels, filter_size_im, filter_size_mask)
    plot_images([im1, im2, mask, im_blend])
    return im1, im2, mask, im_blend


def blending_example2():
    """
    performs pyramid blending and returns:
        im1 and im2 - the images from the blending
        mask - the binary mask
        im_blend - the result
    """
    max_levels = 10
    filter_size_im = 9
    filter_size_mask = 11
    im1 = read_image(relpath("externals/bg2.jpg"), RGB_CODE) # the background img
    im2 = read_image(relpath("externals/obj2.jpg"), RGB_CODE) # the object-masked img
    mask = np.round(read_image(relpath("externals/mask2.jpg"), # the mask
                               GREYSCALE_CODE)).astype(np.bool)
    im_blend = np.zeros((im1.shape))
    for i in range(3):
        im_blend[:,:,i] = pyramid_blending(im1[:,:,i], im2[:,:,i],
                                           mask, max_levels, filter_size_im, filter_size_mask)
    plot_images([im1, im2, mask, im_blend])
    return im1, im2, mask, im_blend


if __name__ == '__main__':
    blending_example1()
    blending_example2()