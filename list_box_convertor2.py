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

def start():
    parser = argparse.ArgumentParser(description='A data convertor intented to resize and add Gaussian border to images and their centroids lists for Cryo-EM datasets.')
    # parser.add_argument('-m', '--mode', required=True, type=str, choices=['list_to_box', 'box_to_list'], dest='mode')
    parser.add_argument('-t', '--image_type', type=str, choices=['tif', 'jpg'], default='tif', dest='image_type')
    # parser.add_argument('-n', '--name', required=True, type=str, dest='name')
    parser.add_argument('-s', '--range_start', type=int, default=1, dest='range_start')
    parser.add_argument('-e', '--range_end', type=int, default=2, dest='range_end')
    parser.add_argument('-b', '--border_size', type=int, default=0, dest='border_size')
    parser.add_argument('-d', '--drive', type=str, default='ssd', choices=['ssd', 'hdd'], dest='drive')
    parser.add_argument('-tw', '--target_width', type=int, default=0, dest='target_width')
    parser.add_argument('-th', '--target_height', type=int, default=0, dest='target_height')
    parser.add_argument('-sl', '--source_list', type=str, default='labels11_2', dest='source_list')
    parser.add_argument('-dl', '--dest_list', type=str, default='labels11_2_cryolo1', dest='dest_list')
    parser.add_argument('-si', '--source_image', type=str, default='trainingdata11_2', dest='source_image')
    parser.add_argument('-di', '--dest_image', type=str, default='trainingdata11_3', dest='dest_image')
    parser.add_argument('-wb', '--write_box', type=int, default=0, dest='write_box')
    args = parser.parse_args()

    # mode = args.mode
    image_type = args.image_type
    # name = args.name
    range_start = args.range_start
    range_end = args.range_end
    border_size = args.border_size
    target_width = args.target_width
    target_height = args.target_height
    source_list = args.source_list
    dest_list = args.dest_list
    source_image = args.source_image
    dest_image = args.dest_image
    drive = args.drive
    write_box = args.write_box

    # if mode == "list":
    path3 = "data/boxnet/" + source_list + "/"
    path4 = "data/boxnet/" + dest_list + '/'
    with open(path3 + 'uri_list.pickle', 'rb') as handle:
        uri_list = pickle.load(handle)
    # with open("data/boxnet/labels4/" + 'uri_list.pickle', 'rb') as handle:
    #     uri_list2 = pickle.load(handle)
    # with open(path3 + 'centroids_list5.pickle', 'rb') as handle:
    #     centroids_list = pickle.load(handle)
    with open("data/boxnet/labels11_2/" + 'centroids_list.pickle', 'rb') as handle:
        centroids_list2 = pickle.load(handle)
    # with open(path3 + 'std_list.pickle', 'rb') as handle:
    #     std_list = pickle.load(handle)
    # with open(path3 + 'avg_list.pickle', 'rb') as handle:
    #     avg_list = pickle.load(handle)

    if drive == "hdd":
        path1 = "data/boxnet/" + source_image + "/"
        # path2 = "data/boxnet/" + dest_image + "/"
    elif drive == "ssd":
        path1 = "/local/ssd/abbasmz/boxnet/" + source_image + "/"
        # path2 = "/local/ssd/abbasmz/boxnet/" + dest_image + "/"

    # for c1 in range(len(uri_list)):
    for c1 in range(1, 2):

        source_path = os.path.join(path1 + uri_list[c1])
        I = open_image(source_path)
        I = I.squeeze()
        source_width, source_height = I.shape

        centroids = centroids_list2[c1][1:].split(' ')

        write_box_file(centroids, len(centroids), 21, path4 + uri_list[c1][:-3] + "box", source_height)

        print("List " + str(c1) + " done.")

    sys.exit(0)

def write_box_file(local_points_list, local_points_number, local_window_size, file_path, source_height):
    # Creates a new or empties the existing local box file and fills it with the list of points.

    # (Re)create empty points file
    box_file = open(file_path, "w+")
    half_window_size = local_window_size / 2
    # Write box file
    i = 0
    while i < local_points_number:
        box_file.write(str((float(local_points_list[i + 1]) - half_window_size)) + "    " + str(
            source_height - (float(local_points_list[i]) - half_window_size)) + "    " + str(
            local_window_size) + "     " + str(local_window_size) + "     " + "-3\n")
        i += 2

    # Close the file
    box_file.close()

def write_star_file(local_points_list, local_points_number, local_window_size, id_number, micrograph_counter):
    # Creates a new or empties the existing local box file and fills it with the list of points.

    # (Re)create empty points file
    star_file = open("imgdata_" + str(id_number) + "_" + str(micrograph_counter) + "_automatch.box", "w+")

    # Write box file
    for i in range(0, local_points_number):
        star_file.write(str(local_points_list[i][0]) + "    " + str(local_points_list[i][1]) + "    " + str(
            local_window_size) + "     " + str(local_window_size) + "     " + "-3\n")

    # Close the file
    star_file.close()

def main(argv=None):
    start()


if __name__ == '__main__':
    main()
