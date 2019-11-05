import csv
from fastai.imports import *
from fastai.torch_imports import *
from fastai.core import *
from fastai.transforms import *
from fastai.layer_optimizer import *
from fastai.dataloader import DataLoader
from fastai import tifffile
import os, sys, pickle # for data serialization (to and from long string of bytes)
import numpy as np
from copy import deepcopy as dc
from scipy import fftpack
import pylab as py
import argparse
from fastai.dataset import open_image

def start():
    parser = argparse.ArgumentParser(description='A data convertor intented to resize and add Gaussian border to images and their centroids lists for Cryo-EM datasets.')
    parser.add_argument('-m', '--mode', required=True, type=str, choices=['image', 'list'], dest='mode')
    parser.add_argument('-t', '--image_type', type=str, choices=['tif', 'jpg'], default='tif', dest='image_type')
    parser.add_argument('-n', '--name', required=True, type=str, dest='name')
    parser.add_argument('-s', '--range_start', type=int, default=1, dest='range_start')
    parser.add_argument('-e', '--range_end', type=int, default=2, dest='range_end')
    parser.add_argument('-b', '--border_size', type=int, default=0, dest='border_size')
    parser.add_argument('-tw', '--target_width', type=int, default=0, dest='target_width')
    parser.add_argument('-th', '--target_height', type=int, default=0, dest='target_height')
    parser.add_argument('-sl', '--source_list', type=str, default='labels6', dest='source_list')
    parser.add_argument('-dl', '--dest_list', type=str, default='labels6', dest='dest_list')
    parser.add_argument('-si', '--source_image', type=str, default='trainingdata5', dest='source_image')
    parser.add_argument('-di', '--dest_image', type=str, default='trainingdata6', dest='dest_image')
    args = parser.parse_args()

    mode = args.mode
    image_type = args.image_type
    name = args.name
    range_start = args.range_start
    range_end = args.range_end
    border_size = args.border_size
    target_width = args.target_width
    target_height = args.target_height
    source_list = args.source_list
    dest_list = args.dest_list
    source_image = args.source_image
    dest_image = args.dest_image

    # if mode == "list":
    path3 = "data/boxnet/" + source_list + "/"
    path4 = "data/boxnet/" + dest_list + '/'
    with open(path3 + 'uri_list5.pickle', 'rb') as handle:
        uri_list = pickle.load(handle)
    with open(path3 + 'centroids_list.pickle', 'rb') as handle:
        centroids_list = pickle.load(handle)
    with open(path3 + 'std_list.pickle', 'rb') as handle:
        std_list = pickle.load(handle)
    with open(path3 + 'avg_list.pickle', 'rb') as handle:
        avg_list = pickle.load(handle)

    path1 = "data/boxnet/" + source_image + "/"
    path2 = "data/boxnet/" + dest_image + "/"

    for c1 in range(range_start, range_end):
        source_path = os.path.join(path1 + name + '_' + str(c1) + '.' + image_type)
        I = open_image(source_path)
        I = I.squeeze()
        source_width, source_height = I.shape

        if target_width == 0:
            target_width = source_width
        if target_height == 0:
            target_height = source_height

        if mode == 'image':

            # Take the fourier transform of the image.
            I_fft = fftpack.fft2(I)

            # Shift the quadrants around so that low spatial frequencies are in
            # the ecenter of the 2D fourier transformed image.
            I_fft_shift = fftpack.fftshift(I_fft)

            if (target_width >= source_width) and (target_height >= source_height):
                target_blank = np.zeros((target_width, target_height), dtype=np.complex64)
                target_blank[(target_width - source_width) // 2: (target_width + source_width) // 2,
                    (target_height - source_height) // 2: (target_height + source_height) // 2] = I_fft_shift
                I_fft_shift_resize = target_blank
            elif (target_width < source_width) and (target_height < source_height):
                I_fft_shift_resize = I_fft_shift[(source_width - target_width) // 2: (source_width + target_width) // 2,
                                     (source_height - target_height) // 2: (source_height + target_height) // 2]
            else:
                print("Error in ratio ?!")
                exit(1)

            I_fft_resize = fftpack.ifftshift(I_fft_shift_resize)

            I_resize = np.real(fftpack.ifft2(I_fft_resize))
            # I_resize = np.expand_dims(I_resize, axis=2)

            source_index = uri_list.index(name + '_' + str(c1) + '.' + image_type)
            org_mean = avg_list[source_index]
            org_std = std_list[source_index]

            blank_width = target_width + 2 * border_size
            blank_height = target_height + 2 * border_size
            target_blank = np.random.normal(org_mean, org_std, (blank_width, blank_height)).astype(np.float32)
            target_blank[border_size: blank_width - border_size, border_size: blank_height - border_size] = I_resize
            I_resize_border = target_blank
            target_path = os.path.join(path2 + name + '_' + str(c1) + '.' + image_type)

            tifffile.imwrite(target_path, I_resize_border)

            print("Image " + str(c1) + " done.")
        else:  # if mode == 'list':
            source_index = uri_list.index(name + '_' + str(c1) + '.' + image_type)
            centroids = centroids_list[source_index][1:].split(' ')

            width_ratio = (target_width) / (source_width * 1.0)
            height_ratio = (target_height) / (source_height * 1.0)

            centroids2 = ''
            for c2 in range(len(centroids)):
                if c2 % 2.0 > 0:  # height
                    centroids2 = ' '.join([centroids2, str(int(np.floor(int(centroids[c2]) * height_ratio) + border_size))])
                else:  # width
                    centroids2 = ' '.join([centroids2, str(int(np.floor(int(centroids[c2]) * width_ratio)) + border_size)])

            centroids_list[source_index] = centroids2
            print("List " + str(c1) + " done.")

    if mode == 'list':
        with open(path4 + 'centroids_list.pickle', 'wb') as handle:
            pickle.dump(centroids_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.exit(0)

def main(argv=None):
    start()


if __name__ == '__main__':
    main()
