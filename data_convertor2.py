# import csv
# from fastai.imports import *
# from fastai.torch_imports import *
# from fastai.core import *
# from fastai.transforms import *
# from fastai.layer_optimizer import *
# from fastai.dataloader import DataLoader
from fastai import tifffile
import os, sys, pickle # for data serialization (to and from long string of bytes)
import numpy as np
from copy import deepcopy as dc
from scipy import fftpack
# import pylab as py
import argparse
from fastai.dataset import open_image
from scipy import ndimage

def start():
    parser = argparse.ArgumentParser(description='A data convertor intented to resize and add Gaussian border to images and their centroids lists for Cryo-EM datasets.')
    parser.add_argument('-m', '--mode', required=True, type=str, choices=['image', 'list'], dest='mode')
    parser.add_argument('-t', '--image_type', type=str, choices=['tif', 'jpg'], default='tif', dest='image_type')
    parser.add_argument('-n', '--name', required=True, type=str, dest='name')
    parser.add_argument('-s', '--range_start', type=int, default=1, dest='range_start')
    parser.add_argument('-e', '--range_end', type=int, default=2, dest='range_end')
    parser.add_argument('-b', '--border_size', type=int, default=0, dest='border_size')
    parser.add_argument('-d', '--drive', type=str, default='hdd', choices=['ssd', 'hdd'], dest='drive')
    parser.add_argument('-tw', '--target_width', type=int, default=0, dest='target_width')
    parser.add_argument('-th', '--target_height', type=int, default=0, dest='target_height')
    parser.add_argument('-sl', '--source_list', type=str, default='labels8', dest='source_list')
    parser.add_argument('-dl', '--dest_list', type=str, default='labels8', dest='dest_list')
    parser.add_argument('-si', '--source_image', type=str, default='trainingdata7', dest='source_image')
    parser.add_argument('-di', '--dest_image', type=str, default='trainingdata8', dest='dest_image')
    args = parser.parse_args()

    mode = args.mode
    image_type = args.image_type
    name = args.name
    range_start = args.range_start
    range_end = args.range_end
    border_size = args.border_size
    # orig_width = args.orig_width
    # orig_height = args.orig_height
    target_width = args.target_width
    target_height = args.target_height
    source_list = args.source_list
    dest_list = args.dest_list
    source_image = args.source_image
    dest_image = args.dest_image
    drive = args.drive

    # if mode == "list":
    path3 = "data/boxnet/" + source_list + "/"
    path4 = "data/boxnet/" + dest_list + '/'
    with open("data/boxnet/labels5/" + 'uri_list5.pickle', 'rb') as handle:
        uri_list = pickle.load(handle)
    with open("data/boxnet/labels4/" + 'uri_list.pickle', 'rb') as handle:
        uri_list2 = pickle.load(handle)
    with open(path3 + 'centroids_list.pickle', 'rb') as handle:
        centroids_list = pickle.load(handle)
    with open("data/boxnet/labels5/" + 'centroids_list5.pickle', 'rb') as handle:
        centroids_list2 = pickle.load(handle)
    with open("data/boxnet/labels5/" + 'std_list.pickle', 'rb') as handle:
        std_list = pickle.load(handle)
    with open("data/boxnet/labels5/" + 'avg_list.pickle', 'rb') as handle:
        avg_list = pickle.load(handle)

    if drive == "hdd":
        path1 = "data/boxnet/" + source_image + "/"
        path2 = "data/boxnet/" + dest_image + "/"
    elif drive == "ssd":
        path1 = "/local/ssd/abbasmz/boxnet/" + source_image + "/"
        path2 = "/local/ssd/abbasmz/boxnet/" + dest_image + "/"

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

            source_index = uri_list.index(name + '_' + str(c1) + '.' + image_type)
            org_mean = avg_list[source_index]
            org_std = std_list[source_index]

            blank_width = target_width + 2 * border_size
            blank_height = target_height + 2 * border_size
            target_blank = np.random.normal(org_mean, org_std, (blank_width, blank_height)).astype(np.float32)

            target_blank[border_size: blank_width - border_size, border_size: blank_height - border_size] = I
            I_resize_border = target_blank
            target_path = os.path.join(path2 + name + '_' + str(c1) + '.' + image_type)

            tifffile.imwrite(target_path, I_resize_border)

            print("Image " + str(c1) + " done.")
        else:  # if mode == 'list':
            source_index = uri_list.index(name + '_' + str(c1) + '.' + image_type)

            centroids = centroids_list2[source_index][1:].split(' ')

            width_ratio = (target_width) / (source_width * 1.0)
            height_ratio = (target_height) / (source_height * 1.0)

            centroids2 = ''
            for c2 in range(len(centroids)):
                if c2 % 2.0 > 0:  # height
                    centroids2 = ' '.join([centroids2, str(int(np.floor(int(centroids[c2]) * height_ratio) + border_size))])
                else:  # width
                    centroids2 = ' '.join([centroids2, str(int(np.floor(int(centroids[c2]) * width_ratio)) + border_size)])

            target_index = uri_list2.index(name + '_' + str(c1) + '.' + image_type)
            centroids_list[target_index] = centroids2
            print("List " + str(c1) + " done.")

    if mode == 'list':
        with open(path4 + 'centroids_list.pickle', 'wb') as handle:
            pickle.dump(centroids_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sys.exit(0)

def main(argv=None):
    start()


if __name__ == '__main__':
    main()
