import sys, os
import pickle
import imageio
import numpy as np
from skimage import measure

# sys.path.append('../')
from fastai.conv_learner import *
from fastai.dataset import *
from fastai import tifffile

from libtiff import TIFF

from pathlib import Path
import json
# from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects

import time
from copy import deepcopy as dc

# from optparse import OptionParser

import argparse

import functools

def start():

    path1='data/boxnet/trainingdata/'
    files_list = os.listdir(path1)
    path2 = 'data/boxnet/trainingdata5/'
    path3 = 'data/boxnet/labels5/'

    centroids_list = []
    classes_lists = []
    uri_list = []
    avg_list = []
    std_list = []
    for file in files_list:

        # to open a tiff file for reading:
        tif = TIFF.open(path1 + file, mode='r')
        # to read an image in the currect TIFF directory and return it as numpy array:
        # image = tif.read_image()
        # to read all images in a TIFF file:
        i = 0
        for img in tif.iter_images():
            print(file + '_' + str(i))
            # do stuff with image
            if i % 3 == 0:
                uri = file[:-4] + '_' + str((i + 3) // 3) + '.tif'
                uri_list.append(uri)
                uri = path2 + uri
                tif = TIFF.open(uri, mode='w')
                tif.write_image(img)
                img1 = img

            elif i % 3 == 1:
                npimg = np.asarray(img, dtype=np.uint8)

                avg_list.append(np.mean(img1[npimg == 0]))
                std_list.append(np.std(img1[npimg == 0]))

                labels = measure.label(npimg)

                regions = measure.regionprops(label_image=labels)

                centroids = ''
                classes = []
                convex_areas = 0

                for region in regions:
                    convex_areas += region.convex_area
                convex_areas /= len(regions)

                for region in regions:
                    centroids = ' '.join([centroids, str(int(region.centroid[0])), str(int(region.centroid[1]))])
                    if region.convex_area < convex_areas * 2:
                        classes.append(0)
                    else:
                        classes.append(1)
                classes_lists.append(classes)
                centroids_list.append(centroids)

            i += 1

    with open(path3+'uri_list5.pickle', 'wb') as handle:
        pickle.dump(uri_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path3+'centroids_list5.pickle', 'wb') as handle:
        pickle.dump(centroids_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path3+'avg_list.pickle', 'wb') as handle:
        pickle.dump(avg_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path3+'std_list.pickle', 'wb') as handle:
        pickle.dump(std_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(argv=None):
    start()
    exit(0)


if __name__ == '__main__':
    main()