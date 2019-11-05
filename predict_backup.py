import sys, os
import pickle
import imageio
import numpy as np
from skimage import measure

# sys.path.append('../')
from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
from matplotlib import patches, patheffects

import time
from copy import deepcopy as dc

import argparse

import functools

from fastai.resnet_29_6 import resnet3 as myresnet

def start():
    # check to make sure you set the device
    # cuda_id = 0
    # torch.cuda.set_device(cuda_id)
    version = '29_6'

    parser = argparse.ArgumentParser(description='A cross dataset generalization study using 37 Cryo-EM datasets.')
    #parser.add_argument('-m', '--mode', required=True, type=str,
    #                    choices=['lrfind', 'train', 'test', 'set5weights', 'keepImprovedHeads', 'predict', 'evaluate',
    #                             'draw'], dest='mode')
    parser.add_argument('-t', '--training_type', required=True, type=str,
                        choices=['2b', '3b', '4b', '5b', '3c', '4c', '5c'], dest='training_type')
    parser.add_argument('-o', '--optimizer_type', default='adam_sgdr', type=str,
                        choices=['adam_sgdr', 'adam', 'sgd'], dest='optimizer_type')
    parser.add_argument('-e', '--epochs', default=250, type=int, dest='epochs')
    parser.add_argument('-c', '--cycle_len', default=40, type=int, dest='cycle_len')
    parser.add_argument('-g', '--gen_multiplier', type=float, default=0, dest='gen_multiplier')
    parser.add_argument('-s', '--save', type=str, default='0', dest='save')
    parser.add_argument('-l', '--load', type=str, default=None, dest='load')
    parser.add_argument('-lv', '--load_version', type=str, default=version, dest='load_version')
    parser.add_argument('-cf', '--convert_from', type=str,
                        choices=['2b', '3b', '4b', '5b', '3c', '4c', '5c'], default=None, dest='convert_from')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, dest='lr')
    parser.add_argument('-wd', '--weight_decay', type=float, default=None, dest='wd')
    parser.add_argument('-uwds', '--use_weight_decay_scheduler', type=bool, default=False, dest='uwds')
    parser.add_argument('-lr0', '--learning_rate_0', type=float, default=None, dest='lr0')
    parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=1e-3, dest='lr_decay')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, dest='bs')
    parser.add_argument('-sd', '--source_datasets',
                        default='10077$10078$10081$pdb_6bqv$ss_1$pdb_6bhu$pdb_6bcx$pdb_6bcq$pdb_6bco$pdb_6az1$pdb_5y6p$pdb_5xwy$pdb_5w3s$pdb_5ngm$pdb_5mmi$pdb_5foj$pdb_4zor$pdb_3j9i$pdb_2gtl$pdb_1sa0$lf_1$hh_2$gk_1$10156$10153$10122$10097$10089$10084$10017',
                        type=str, dest='source_dataset')
    parser.add_argument('-td', '--target_datasets',
                        default='pdb_6b7n$pdb_6b44$pdb_5xnl$pdb_5w3l$pdb_5vy5$pdb_4hhb$pdb_2wri', type=str,
                        dest='target_dataset')
    parser.add_argument('-cs', '--crop_size', type=int, default=368, dest='crop_size')  # up to 240 for original images
    parser.add_argument('-ps', '--particle_size', type=int, default=1, dest='particle_size')
    parser.add_argument('-si', '--source_image', type=str, default='trainingdata11_2', dest='source_image')
    parser.add_argument('-sl', '--source_list', type=str, default='labels12_2', dest='source_list')
    parser.add_argument('-d', '--drive', default='ssd', type=str, choices=['ssd', 'hdd'], dest='drive')
    parser.add_argument('-ioutrn', '--IOU_training', type=float, default=0.6, dest='iou_trn')
    parser.add_argument('-ioutst', '--IOU_testing', type=float, default=0.6, dest='iou_tst')
    parser.add_argument('-ho', '--heads_only', type=int, default=0, choices=[0, 1], dest='heads_only')
    parser.add_argument('-tgh', '--train_gen_head', type=int, default=1, choices=[0, 1], dest='train_gen_head')
    parser.add_argument('-tfth', '--train_fine_tune_head', type=int, default=0, choices=[0, 1], dest='train_fine_tune_head')
    parser.add_argument('-cp', '--check_pointing', type=int, default=0, choices=[0, 1], dest='check_pointing')
    parser.add_argument('-ph', '--prediction_head', type=str, default='gen', dest='prediction_head')
    parser.add_argument('-psf', '--prediction_subfolder', type=str, default=None, dest='prediction_subfolder')
    parser.add_argument('-pc', '--prediction_conf', type=float, default=0, dest='prediction_conf')
    parser.add_argument('-et', '--evaluate_type', type=str, choices=['sources', 'targets', 'folder'],
                        default='targets', dest='evaluate_type')
    parser.add_argument('-ef', '--evaluate_format', type=str, choices=['csv', 'star', 'star_cryolo', 'box', 'box_y_inversed'],
                        default='csv', dest='evaluate_format')
    parser.add_argument('-efs', '--evaluate_fscore', type=int, choices=[0, 1], default=0, dest='evaluate_fscore')
    parser.add_argument('-cm', '--cryolo_model', type=str, default='phosnet', dest='cryolo_model')
    parser.add_argument('-bns', '--boxnet_suffix', type=str, default='_BoxNet_blank_trn5.star', dest='boxnet_suffix')
    args = parser.parse_args()


    ############ Mainly used variables
    parameters_text = ''

    #mode = args.mode
    #parameters_text += "mode: " + mode

    training_type = args.training_type
    parameters_text += "\ntraining_type: " + training_type

    optimizer_type = args.optimizer_type
    parameters_text += "\noptimizer_type: " + optimizer_type

    epochs = args.epochs
    parameters_text += "\nepochs: " + str(epochs)

    cycle_len = args.cycle_len
    parameters_text += "\ncycle_len: " + str(cycle_len)

    gen_multiplier = args.gen_multiplier
    parameters_text += "\ngen_multiplier: " + str(gen_multiplier)

    save = args.save
    parameters_text += "\nsave: " + save

    load = args.load
    if load:
        parameters_text += "\nload: " + load
    else:
        parameters_text += "\nload: " + "None"

    load_version = args.load_version
    parameters_text += "\nload_version: " + load_version

    convert_from = args.convert_from
    if convert_from:
        parameters_text += "\nconvert_from: " + convert_from
    else:
        parameters_text += "\nconvert_from: " + "None"

    source_image = args.source_image
    parameters_text += "\nsource_image: " + source_image

    source_list = args.source_list
    parameters_text += "\nsource_list: " + source_list

    if args.lr0 is None:
        lr = args.lr
        parameters_text += "\nlearning_rate: " + str(lr)
    else:
        lr = (args.lr0, args.lr0, args.lr)
        parameters_text += "\nlearning_rate: " + str(lr) + ", learning_rate_0: "  + str(args.lr0)

    uwds = args.uwds
    parameters_text += "\nuse_weight_decay_scheduler: " + str(uwds)

    wd = args.wd
    parameters_text += "\nweight_decay: " + str(wd)

    lr_decay = args.lr_decay
    parameters_text += "\nlr_decay: " + str(lr_decay)

    bs = args.bs
    parameters_text += "\nbatch_size: " + str(bs)

    source_datasets = args.source_dataset.split('$')
    parameters_text += "\nsource_datasets: " + args.source_dataset

    target_datasets = args.target_dataset.split('$')
    parameters_text += "\ntarget_datasets: " + args.target_dataset

    iou_trn = args.iou_trn
    parameters_text += "\nIOU_training: " + str(iou_trn)

    iou_tst = args.iou_tst
    parameters_text += "\nIOU_testing: " + str(iou_tst)

    f_model = myresnet
    sz = args.crop_size
    parameters_text += "\ncrop_size: " + str(sz)

    # particle_size_dict_0 = {"10077": 25, "10078": 20, "10081": 18, "pdb_6bqv": 17, "ss_1": 20, "pdb_6bhu": 14,
    #                         "pdb_6bcx": 23, "pdb_6bcq": 26, "pdb_6bco": 15, "pdb_6az1": 32, "pdb_5y6p": 78,
    #                         "pdb_5xwy": 11, "pdb_5w3s": 13, "pdb_5ngm": 30, "pdb_5mmi": 24, "pdb_5foj": 8,
    #                         "pdb_4zor": 17, "pdb_3j9i": 18, "pdb_2gtl": 13, "pdb_1sa0": 20, "lf_1": 15, "hh_2": 21,
    #                         "gk_1": 23, "10156": 31, "10153": 25, "10122": 16, "10097": 14, "10089": 24, "10084": 8,
    #                         "10017": 22, "pdb_6b7n": 14, "pdb_6b44": 17, "pdb_5xnl": 37, "pdb_5w3l": 38, "pdb_5vy5": 10,
    #                         "pdb_4hhb": 7, "pdb_2wri": 24}
    # particle_size_dict_1 = {"10077": 20, "10078": 20, "10081": 18, "pdb_6bqv": 17, "ss_1": 20, "pdb_6bhu": 21,
    #                         "pdb_6bcx": 23, "pdb_6bcq": 21, "pdb_6bco": 21, "pdb_6az1": 24, "pdb_5y6p": 21,
    #                         "pdb_5xwy": 22, "pdb_5w3s": 21, "pdb_5ngm": 21, "pdb_5mmi": 24, "pdb_5foj": 24,
    #                         "pdb_4zor": 17, "pdb_3j9i": 18, "pdb_2gtl": 21, "pdb_1sa0": 20, "lf_1": 21, "hh_2": 21,
    #                         "gk_1": 23, "10156": 21, "10153": 23, "10122": 24, "10097": 21, "10089": 24, "10084": 24,
    #                         "10017": 22, "pdb_6b7n": 21, "pdb_6b44": 17, "pdb_5xnl": 22, "pdb_5w3l": 19, "pdb_5vy5": 20,
    #                         "pdb_4hhb": 21, "pdb_2wri": 24}
    if args.particle_size == 0:
        # par_sz_pix = particle_size_dict_0[target_dataset]
        par_sz_pix = 20
    elif args.particle_size == 1:
        # par_sz_pix = particle_size_dict_1[target_dataset]
        par_sz_pix = 21
    else:
        par_sz_pix = args.particle_size
    parameters_text += "\nparticle_size_pixels: " + str(par_sz_pix)

    par_sz = par_sz_pix / sz
    parameters_text += "\nparticle_size_ratio: " + str(par_sz)

    if training_type == "4b" or training_type == "5b" or training_type == "4c" or training_type == "5c":
        unbiased_training = True
    else:
        unbiased_training = False
    parameters_text += "\nunbiased_training: " + str(unbiased_training)

    heads_only = args.heads_only
    # auxilary.auxilary.heads_training_only[0] = heads_only * torch.eye(1, dtype=torch.int8)
    parameters_text += "\nheads_training_only: " + str(heads_only)

    check_pointing = args.check_pointing
    parameters_text += "\ncheck_pointing: " + str(check_pointing)

    # dist_threshold = par_sz * 0.2
    # pick_threshold = np.linspace(1e-6, 1 - 1e-6, 41)
    heads_dict = {"10077": 0, "10078": 1, "10081": 2, "pdb_6bqv": 3, "ss_1": 4,
                  "pdb_6bhu": 5, "pdb_6bcx": 6, "pdb_6bcq": 7, "pdb_6bco": 8,
                  "pdb_6az1": 9, "pdb_5y6p": 10, "pdb_5xwy": 11, "pdb_5w3s": 12,
                  "pdb_5ngm": 13, "pdb_5mmi": 14, "pdb_5foj": 15, "pdb_4zor": 16,
                  "pdb_3j9i": 17, "pdb_2gtl": 18, "pdb_1sa0": 19, "lf_1": 20,
                  "hh_2": 21, "gk_1": 22, "10156": 23, "10153": 24, "10122": 25,
                  "10097": 26, "10089": 27, "10084": 28, "10017": 29,
                  "pdb_6b7n": 30, "pdb_6b44": 31, "pdb_5xnl": 32, "pdb_5w3l": 33,
                  "pdb_5vy5": 34, "pdb_4hhb": 35, "pdb_2wri": 36, "gen": 37, "fine_tuned": 38}

    train_gen_head = args.train_gen_head
    parameters_text += "\nTrain_gen_head: " + str(train_gen_head)

    train_fine_tune_head = args.train_fine_tune_head
    parameters_text += "\nTrain_fine_tune_head: " + str(train_fine_tune_head)

    if training_type == '3c' or training_type == '4c' or training_type == '5c':
        # auxilary.auxilary.fine_tune[0] = 1 * torch.eye(1, dtype=torch.int8)
        auxilary.auxilary.fine_tune[0] = len(target_datasets) * torch.eye(1, dtype=torch.int8)
        for i in range(1, len(target_datasets) + 1):
            auxilary.auxilary.fine_tune[i] = heads_dict[target_datasets[i - 1]] * torch.eye(1, dtype=torch.int8)
        auxilary.auxilary.fine_tune_wgen[0] = train_gen_head * torch.eye(1, dtype=torch.int8)
        auxilary.auxilary.fine_tune_wgen[1] = heads_dict["gen"] * torch.eye(1, dtype=torch.int8)
        auxilary.auxilary.fine_tune_wfine[0] = train_fine_tune_head * torch.eye(1, dtype=torch.int8)
        auxilary.auxilary.fine_tune_wfine[1] = heads_dict["fine_tuned"] * torch.eye(1, dtype=torch.int8)
    else:
        auxilary.auxilary.fine_tune[0] = 0 * torch.eye(1, dtype=torch.int8)

    prediction_head = args.prediction_head
    parameters_text += "\nprediction_head: " + prediction_head

    prediction_subfolder = args.prediction_subfolder
    if prediction_subfolder:
        parameters_text += "\nprediction_subfolder: " + prediction_subfolder
    else:
        parameters_text += "\nprediction_subfolder: None"

    prediction_conf = args.prediction_conf
    parameters_text += "\nprediction_conf: " + str(prediction_conf)

    evaluate_type = args.evaluate_type
    parameters_text += "\nevaluate_type: " + evaluate_type

    evaluate_format = args.evaluate_format
    parameters_text += "\nevaluate_format: " + evaluate_format

    evaluate_fscore = args.evaluate_fscore
    parameters_text += "\nevaluate_fscore: " + str(evaluate_fscore)

    cryolo_model = args.cryolo_model
    parameters_text += "\ncryolo_model: " + str(cryolo_model)

    boxnet_suffix = args.boxnet_suffix
    parameters_text += "\nboxnet_suffix: " + str(boxnet_suffix)

    detections = []

    ############ Useful paths
    drive = args.drive
    parameters_text += "\ndrive: " + str(drive)
    if drive == 'ssd':
        path2 = "/local/ssd/abbasmz/boxnet/" + source_image + "/"
        PATH = Path("/local/ssd/abbasmz/boxnet")
    else:
        path2 = "data/boxnet/" + source_image + "/"
        PATH = Path("data/boxnet")
    PATH2 = Path("data/boxnet")
    path3 = "data/boxnet/" + source_list + "/"
    path4 = "data/boxnet/results/"
    IMAGES = source_image


    ############
    # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects#31174427
    # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    def rsetattr(obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(rgetattr(obj, pre) if pre else obj, post, val)

    def rgetattr(obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split('.'))

    with open(path3+'uri_list.pickle', 'rb') as handle:
        uri_list = pickle.load(handle)

    with open(path3+'centroids_list.pickle', 'rb') as handle:
        centroids_list = pickle.load(handle)

    rand_10077 = [16, 14, 0, 6, 8, 21, 22, 5, 18, 11, 10, 1, 2, 15, 9, 24, 19, 13, 7, 12, 20, 23, 3, 25, 17, 4]
    rand_10078 = [40, 42, 55, 52, 26, 34, 54, 33, 38, 43, 49, 46, 31, 50, 32, 36, 47, 53, 39, 28, 48, 29, 27, 45, 30,
                  41, 35, 51, 44, 37]
    rand_10081 = [84, 81, 79, 72, 69, 62, 77, 85, 56, 63, 58, 78, 68, 82, 60, 74, 57, 73, 64, 66, 67, 80, 83, 61, 76,
                  71, 75, 65, 59, 70]
    rand_6bqv = [87, 93, 92, 86, 88, 90, 89, 91]
    rand_ss_1 = [109, 173, 114, 102, 138, 184, 104, 195, 125, 107, 94, 177, 115, 149, 134, 186, 103, 142, 119, 112, 152,
                 188, 158, 129, 193, 187, 153, 172, 110, 160, 127, 191, 96, 147, 141, 111, 124, 165, 166, 174, 181, 183,
                 155, 118, 176, 189, 146, 194, 156, 139, 140, 126, 131, 162, 150, 145, 196, 117, 161, 180, 128, 101,
                 192, 133, 164, 116, 99, 98, 148, 105, 122, 190, 169, 121, 182, 108, 157, 144, 163, 171, 130, 159, 135,
                 170, 100, 95, 113, 185, 179, 123, 167, 136, 132, 143, 151, 120, 178, 168, 137, 154, 106, 175, 97]
    rand_6bhu = [197, 203, 202, 204, 201, 200, 198, 199]
    rand_6bcx = [207, 212, 210, 208, 205, 209, 211, 206]
    rand_6bcq = [218, 215, 220, 214, 217, 216, 213, 219]
    rand_6bco = [228, 221, 222, 223, 225, 224, 226, 227]
    rand_6b7n = [231, 229, 234, 233, 235, 230, 232, 236]
    rand_6b44 = [242, 239, 237, 241, 243, 244, 240, 238]
    rand_6az1 = [245, 250, 248, 249, 247, 252, 251, 246]
    rand_5y6p = [267, 265, 274, 268, 263, 262, 258, 271, 254, 273, 256, 264, 270, 260, 259, 255, 266, 272, 253, 269,
                 261, 257]
    rand_5xwy = [278, 275, 281, 282, 279, 280, 277, 276]
    rand_5xnl = [283, 285, 290, 287, 284, 286, 289, 288]
    rand_5w3s = [293, 298, 296, 294, 291, 292, 297, 295]
    rand_5w3l = [305, 300, 299, 301, 304, 303, 302, 306]
    rand_5vy5 = [310, 307, 309, 311, 308]
    rand_5ngm = [315, 313, 318, 314, 316, 312, 317, 319]
    rand_5mmi = [321, 320, 322, 326, 324, 325, 323, 327]
    rand_5foj = [331, 330, 329, 328, 332]
    rand_4zor = [340, 333, 336, 334, 339, 335, 337, 338]
    rand_4hhb = [343, 344, 342, 341]
    rand_3j9i = [345, 347, 350, 346, 352, 351, 348, 349]
    rand_2wri = [356, 360, 354, 353, 359, 355, 357, 358]
    rand_2gtl = [364, 368, 367, 361, 362, 366, 363, 365]
    rand_1sa0 = [374, 369, 373, 375, 376, 372, 371, 370]
    rand_lf_1 = [379, 384, 378, 381, 380, 377, 382, 385, 383, 386, 387]
    rand_hh_2 = [407, 389, 408, 413, 405, 388, 394, 410, 416, 406, 395, 417, 402, 404, 414, 393, 396, 409, 415, 392,
                 390, 399, 397, 412, 403, 391, 401, 421, 419, 420, 400, 418, 422, 411, 398, 423]
    rand_gk_1 = [431, 436, 439, 441, 437, 440, 428, 430, 426, 427, 434, 429, 424, 432, 438, 435, 433, 425]
    rand_10156 = [462, 459, 444, 446, 461, 447, 451, 443, 454, 448, 455, 460, 452, 445, 457, 450, 449, 442, 458, 453,
                  456]
    rand_10153 = [492, 479, 499, 483, 509, 513, 503, 473, 510, 494, 491, 486, 480, 532, 471, 518, 493, 524, 520, 474,
                  525, 481, 490, 469, 523, 505, 522, 495, 476, 511, 497, 464, 463, 515, 516, 501, 489, 527, 507, 487,
                  472, 468, 500, 488, 498, 514, 519, 521, 530, 526, 496, 478, 529, 475, 504, 467, 484, 528, 477, 485,
                  506, 466, 512, 502, 533, 470, 482, 508, 517, 531, 465]
    rand_10122 = [556, 540, 534, 542, 550, 551, 552, 543, 549, 547, 539, 558, 541, 557, 535, 546, 553, 548, 537, 554,
                  545, 544, 536, 555, 538]
    rand_10097 = [573, 562, 568, 587, 572, 574, 583, 569, 564, 575, 576, 561, 571, 581, 565, 580, 585, 578, 559, 566,
                  588, 582, 584, 563, 579, 560, 586, 567, 577, 570]
    rand_10089 = [600, 606, 603, 590, 604, 611, 592, 605, 593, 597, 596, 599, 598, 602, 591, 594, 612, 595, 609, 610,
                  589, 607, 608, 601]
    rand_10084 = [620, 616, 615, 626, 613, 619, 618, 623, 617, 614, 624, 622, 625, 621, 627]
    rand_10017 = [676, 646, 691, 707, 645, 701, 704, 667, 693, 700, 668, 635, 690, 629, 666, 640, 653, 705, 706, 641,
                  695, 685, 634, 654, 680, 652, 655, 639, 688, 681, 665, 633, 657, 673, 637, 677, 678, 687, 659, 632,
                  689, 683, 709, 671, 663, 658, 699, 686, 628, 647, 702, 698, 631, 703, 674, 692, 642, 710, 670, 650,
                  697, 651, 664, 711, 684, 694, 630, 675, 660, 644, 643, 672, 648, 638, 696, 679, 682, 656, 669, 649,
                  662, 636, 661, 708]

    # rand_10122 = np.array(random.sample(range(26,56), len(range(26,56))))
    # [a[j] for j in rand_10081]


    ######### splitting for each dataset for training
    lens_all_n7 = []
    uri_list10077 = [uri_list[i] for i in rand_10077[3:]]
    lens_all_n7.append(len(uri_list10077))
    uri_list10078 = [uri_list[i] for i in rand_10078[3:]]
    lens_all_n7.append(len(uri_list10078))
    uri_list10081 = [uri_list[i] for i in rand_10081[3:]]
    lens_all_n7.append(len(uri_list10081))
    uri_listpdb_6bqv = [uri_list[i] for i in rand_6bqv[2:]]
    lens_all_n7.append(len(uri_listpdb_6bqv))
    uri_listss_1 = [uri_list[i] for i in rand_ss_1[3:]]
    lens_all_n7.append(len(uri_listss_1))
    uri_listpdb_6bhu = [uri_list[i] for i in rand_6bhu[2:]]
    lens_all_n7.append(len(uri_listpdb_6bhu))
    uri_listpdb_6bcx = [uri_list[i] for i in rand_6bcx[2:]]
    lens_all_n7.append(len(uri_listpdb_6bcx))
    uri_listpdb_6bcq = [uri_list[i] for i in rand_6bcq[2:]]
    lens_all_n7.append(len(uri_listpdb_6bcq))
    uri_listpdb_6bco = [uri_list[i] for i in rand_6bco[2:]]
    lens_all_n7.append(len(uri_listpdb_6bco))
    uri_listpdb_6az1 = [uri_list[i] for i in rand_6az1[2:]]
    lens_all_n7.append(len(uri_listpdb_6az1))
    uri_listpdb_5y6p = [uri_list[i] for i in rand_5y6p[3:]]
    lens_all_n7.append(len(uri_listpdb_5y6p))
    uri_listpdb_5xwy = [uri_list[i] for i in rand_5xwy[2:]]
    lens_all_n7.append(len(uri_listpdb_5xwy))
    uri_listpdb_5w3s = [uri_list[i] for i in rand_5w3s[2:]]
    lens_all_n7.append(len(uri_listpdb_5w3s))
    uri_listpdb_5ngm = [uri_list[i] for i in rand_5ngm[2:]]
    lens_all_n7.append(len(uri_listpdb_5ngm))
    uri_listpdb_5mmi = [uri_list[i] for i in rand_5mmi[2:]]
    lens_all_n7.append(len(uri_listpdb_5mmi))
    uri_listpdb_5foj = [uri_list[i] for i in rand_5foj[1:]]
    lens_all_n7.append(len(uri_listpdb_5foj))
    uri_listpdb_4zor = [uri_list[i] for i in rand_4zor[2:]]
    lens_all_n7.append(len(uri_listpdb_4zor))
    uri_listpdb_3j9i = [uri_list[i] for i in rand_3j9i[2:]]
    lens_all_n7.append(len(uri_listpdb_3j9i))
    uri_listpdb_2gtl = [uri_list[i] for i in rand_2gtl[2:]]
    lens_all_n7.append(len(uri_listpdb_2gtl))
    uri_listpdb_1sa0 = [uri_list[i] for i in rand_1sa0[2:]]
    lens_all_n7.append(len(uri_listpdb_1sa0))
    uri_listlf_1 = [uri_list[i] for i in rand_lf_1[3:]]
    lens_all_n7.append(len(uri_listlf_1))
    uri_listhh_2 = [uri_list[i] for i in rand_hh_2[3:]]
    lens_all_n7.append(len(uri_listhh_2))
    uri_listgk_1 = [uri_list[i] for i in rand_gk_1[3:]]
    lens_all_n7.append(len(uri_listgk_1))
    uri_list10156 = [uri_list[i] for i in rand_10156[3:]]
    lens_all_n7.append(len(uri_list10156))
    uri_list10153 = [uri_list[i] for i in rand_10153[3:]]
    lens_all_n7.append(len(uri_list10153))
    uri_list10122 = [uri_list[i] for i in rand_10122[3:]]
    lens_all_n7.append(len(uri_list10122))
    uri_list10097 = [uri_list[i] for i in rand_10097[3:]]
    lens_all_n7.append(len(uri_list10097))
    uri_list10089 = [uri_list[i] for i in rand_10089[3:]]
    lens_all_n7.append(len(uri_list10089))
    uri_list10084 = [uri_list[i] for i in rand_10084[3:]]
    lens_all_n7.append(len(uri_list10084))
    uri_list10017 = [uri_list[i] for i in rand_10017[3:]]
    lens_all_n7.append(len(uri_list10017))

    lens_all = dc(lens_all_n7)
    uri_listpdb_6b7n = [uri_list[i] for i in rand_6b7n[2:]]
    lens_all.append(len(uri_listpdb_6b7n))
    uri_listpdb_6b44 = [uri_list[i] for i in rand_6b44[2:]]
    lens_all.append(len(uri_listpdb_6b44))
    uri_listpdb_5xnl = [uri_list[i] for i in rand_5xnl[2:]]
    lens_all.append(len(uri_listpdb_5xnl))
    uri_listpdb_5w3l = [uri_list[i] for i in rand_5w3l[2:]]
    lens_all.append(len(uri_listpdb_5w3l))
    uri_listpdb_5vy5 = [uri_list[i] for i in rand_5vy5[1:]]
    lens_all.append(len(uri_listpdb_5vy5))
    uri_listpdb_4hhb = [uri_list[i] for i in rand_4hhb[1:]]
    lens_all.append(len(uri_listpdb_4hhb))
    uri_listpdb_2wri = [uri_list[i] for i in rand_2wri[2:]]
    lens_all.append(len(uri_listpdb_2wri))

    uri_list_dict = {"10077": [uri_list[i] for i in rand_10077[3:]],
                     "10078": [uri_list[i] for i in rand_10078[3:]],
                     "10081": [uri_list[i] for i in rand_10081[3:]],
                     "pdb_6bqv": [uri_list[i] for i in rand_6bqv[2:]],
                     "ss_1": [uri_list[i] for i in rand_ss_1[3:]],
                     "pdb_6bhu": [uri_list[i] for i in rand_6bhu[2:]],
                     "pdb_6bcx": [uri_list[i] for i in rand_6bcx[2:]],
                     "pdb_6bcq": [uri_list[i] for i in rand_6bcq[2:]],
                     "pdb_6bco": [uri_list[i] for i in rand_6bco[2:]],
                     "pdb_6az1": [uri_list[i] for i in rand_6az1[2:]],
                     "pdb_5y6p": [uri_list[i] for i in rand_5y6p[3:]],
                     "pdb_5xwy": [uri_list[i] for i in rand_5xwy[2:]],
                     "pdb_5w3s": [uri_list[i] for i in rand_5w3s[2:]],
                     "pdb_5ngm": [uri_list[i] for i in rand_5ngm[2:]],
                     "pdb_5mmi": [uri_list[i] for i in rand_5mmi[2:]],
                     "pdb_5foj": [uri_list[i] for i in rand_5foj[1:]],
                     "pdb_4zor": [uri_list[i] for i in rand_4zor[2:]],
                     "pdb_3j9i": [uri_list[i] for i in rand_3j9i[2:]],
                     "pdb_2gtl": [uri_list[i] for i in rand_2gtl[2:]],
                     "pdb_1sa0": [uri_list[i] for i in rand_1sa0[2:]],
                     "lf_1": [uri_list[i] for i in rand_lf_1[3:]],
                     "hh_2": [uri_list[i] for i in rand_hh_2[3:]],
                     "gk_1": [uri_list[i] for i in rand_gk_1[3:]],
                     "10156": [uri_list[i] for i in rand_10156[3:]],
                     "10153": [uri_list[i] for i in rand_10153[3:]],
                     "10122": [uri_list[i] for i in rand_10122[3:]],
                     "10097": [uri_list[i] for i in rand_10097[3:]],
                     "10089": [uri_list[i] for i in rand_10089[3:]],
                     "10084": [uri_list[i] for i in rand_10084[3:]],
                     "10017": [uri_list[i] for i in rand_10017[3:]],
                     "pdb_6b7n": [uri_list[i] for i in rand_6b7n[2:]],
                     "pdb_6b44": [uri_list[i] for i in rand_6b44[2:]],
                     "pdb_5xnl": [uri_list[i] for i in rand_5xnl[2:]],
                     "pdb_5w3l": [uri_list[i] for i in rand_5w3l[2:]],
                     "pdb_5vy5": [uri_list[i] for i in rand_5vy5[1:]],
                     "pdb_4hhb": [uri_list[i] for i in rand_4hhb[1:]],
                     "pdb_2wri": [uri_list[i] for i in rand_2wri[2:]]}

    centroids_list_dict = {"10077": [centroids_list[i] for i in rand_10077[3:]],
                           "10078": [centroids_list[i] for i in rand_10078[3:]],
                           "10081": [centroids_list[i] for i in rand_10081[3:]],
                           "pdb_6bqv": [centroids_list[i] for i in rand_6bqv[2:]],
                           "ss_1": [centroids_list[i] for i in rand_ss_1[3:]],
                           "pdb_6bhu": [centroids_list[i] for i in rand_6bhu[2:]],
                           "pdb_6bcx": [centroids_list[i] for i in rand_6bcx[2:]],
                           "pdb_6bcq": [centroids_list[i] for i in rand_6bcq[2:]],
                           "pdb_6bco": [centroids_list[i] for i in rand_6bco[2:]],
                           "pdb_6az1": [centroids_list[i] for i in rand_6az1[2:]],
                           "pdb_5y6p": [centroids_list[i] for i in rand_5y6p[3:]],
                           "pdb_5xwy": [centroids_list[i] for i in rand_5xwy[2:]],
                           "pdb_5w3s": [centroids_list[i] for i in rand_5w3s[2:]],
                           "pdb_5ngm": [centroids_list[i] for i in rand_5ngm[2:]],
                           "pdb_5mmi": [centroids_list[i] for i in rand_5mmi[2:]],
                           "pdb_5foj": [centroids_list[i] for i in rand_5foj[1:]],
                           "pdb_4zor": [centroids_list[i] for i in rand_4zor[2:]],
                           "pdb_3j9i": [centroids_list[i] for i in rand_3j9i[2:]],
                           "pdb_2gtl": [centroids_list[i] for i in rand_2gtl[2:]],
                           "pdb_1sa0": [centroids_list[i] for i in rand_1sa0[2:]],
                           "lf_1": [centroids_list[i] for i in rand_lf_1[3:]],
                           "hh_2": [centroids_list[i] for i in rand_hh_2[3:]],
                           "gk_1": [centroids_list[i] for i in rand_gk_1[3:]],
                           "10156": [centroids_list[i] for i in rand_10156[3:]],
                           "10153": [centroids_list[i] for i in rand_10153[3:]],
                           "10122": [centroids_list[i] for i in rand_10122[3:]],
                           "10097": [centroids_list[i] for i in rand_10097[3:]],
                           "10089": [centroids_list[i] for i in rand_10089[3:]],
                           "10084": [centroids_list[i] for i in rand_10084[3:]],
                           "10017": [centroids_list[i] for i in rand_10017[3:]],
                           "pdb_6b7n": [centroids_list[i] for i in rand_6b7n[2:]],
                           "pdb_6b44": [centroids_list[i] for i in rand_6b44[2:]],
                           "pdb_5xnl": [centroids_list[i] for i in rand_5xnl[2:]],
                           "pdb_5w3l": [centroids_list[i] for i in rand_5w3l[2:]],
                           "pdb_5vy5": [centroids_list[i] for i in rand_5vy5[1:]],
                           "pdb_4hhb": [centroids_list[i] for i in rand_4hhb[1:]],
                           "pdb_2wri": [centroids_list[i] for i in rand_2wri[2:]]}

    lens_dict = {"10077": 3, "10078": 3, "10081": 3, "pdb_6bqv": 2, "ss_1": 3,
                 "pdb_6bhu": 2, "pdb_6bcx": 2, "pdb_6bcq": 2, "pdb_6bco": 2,
                 "pdb_6az1": 2, "pdb_5y6p": 3, "pdb_5xwy": 2, "pdb_5w3s": 2,
                 "pdb_5ngm": 2, "pdb_5mmi": 2, "pdb_5foj": 1, "pdb_4zor": 2,
                 "pdb_3j9i": 2, "pdb_2gtl": 2, "pdb_1sa0": 2, "lf_1": 3,
                 "hh_2": 3, "gk_1": 3, "10156": 3, "10153": 3, "10122": 3,
                 "10097": 3, "10089": 3, "10084": 3, "10017": 3, "pdb_6b7n": 2,
                 "pdb_6b44": 2, "pdb_5xnl": 2, "pdb_5w3l": 2, "pdb_5vy5": 1,
                 "pdb_4hhb": 1, "pdb_2wri": 2}


    # #### Number of all used datasets
    def numlist(count, number): return [number] * count

    num_list_trn = []
    num_list_val = []
    if training_type == "3b" or training_type == "4b" or training_type == "5b":
        for c1 in source_datasets:
            num_list_trn += numlist(len(uri_list_dict[c1]) - lens_dict[c1], heads_dict[c1])
            num_list_val += numlist(lens_dict[c1], heads_dict[c1])
    elif training_type == "3c" or training_type == "4c" or training_type == "5c":
        for c1 in target_datasets:
            num_list_trn += numlist(len(uri_list_dict[c1]) - lens_dict[c1], heads_dict[c1])
            num_list_val += numlist(lens_dict[c1], heads_dict[c1])

    class ConcatLblDataset_trn(Dataset):
        def __init__(self, ds):
            self.ds = ds
            self.sz = ds.sz

        def __len__(self): return len(self.ds)

        def __getitem__(self, i):
            x, y = self.ds[i]
            nonzeros = sum(np.sum(y.reshape(-1,2), 1) > 0)
            # nonzeros = sum(y > 0) // 2
            return (x, (y, np.ones(nonzeros), num_list_trn[i]))


    class ConcatLblDataset_val(Dataset):
        def __init__(self, ds):
            self.ds = ds
            self.sz = ds.sz

        def __len__(self): return len(self.ds)

        def __getitem__(self, i):
            x, y = self.ds[i]
            nonzeros = sum(np.sum(y.reshape(-1,2), 1) > 0)
            # nonzeros = sum(y > 0) // 2
            return (x, (y, np.ones(nonzeros), num_list_val[i]))

    class ConcatLblDataset2(Dataset):
        def __init__(self, ds, num):
            self.ds = ds
            self.sz = ds.sz
            self.num = num

        def __len__(self): return len(self.ds)

        def __getitem__(self, i):
            x, y = self.ds[i]
            nonzeros = sum(np.sum(y.reshape(-1, 2), 1) > 0)
            # nonzeros = sum(y > 0) // 2
            return (x, (y, np.ones(nonzeros), self.num))

    ######## Augmentations
    aug_tfms = [RandomRotate(180, mode=cv2.BORDER_REFLECT_101, tfm_y=TfmType.COORD_CENTERS),
                RandomFlip(tfm_y=TfmType.COORD_CENTERS)]
    # aug_tfms = [RandomRotate(180, mode=cv2.BORDER_REFLECT_101, tfm_y=TfmType.COORD_CENTERS)]
    tfms = tfms_from_model(f_model, sz, crop_type=CropType.LTDCENTER, tfm_y=TfmType.COORD_CENTERS, aug_tfms=aug_tfms,
                           pad_mode=cv2.BORDER_REFLECT_101)
    # tfms = tfms_from_model(f_model, sz, crop_type=CropType.NEARCENTER, tfm_y=TfmType.COORD_CENTERS, aug_tfms=aug_tfms, pad_mode=cv2.BORDER_REFLECT_101)

    source_uri_list = []
    source_centroids_list = []
    source_len_trn = []
    source_len_val = []
    source_val_idxs = ()
    source_val_idxs_index = 0
    for c1 in source_datasets:
        source_uri_list.extend(uri_list_dict[c1])
        source_centroids_list.extend(centroids_list_dict[c1])
        source_len_trn.append(len(uri_list_dict[c1]) - lens_dict[c1])
        source_len_val.append(lens_dict[c1])
        source_val_idxs = source_val_idxs + tuple(range(source_val_idxs_index, source_val_idxs_index + lens_dict[c1]))
        source_val_idxs_index += len(uri_list_dict[c1])

    target_uri_list = []
    target_centroids_list = []
    target_len_trn = []
    target_len_val = []
    target_val_idxs = ()
    target_val_idxs_index = 0
    for c1 in target_datasets:
        target_uri_list.extend(uri_list_dict[c1])
        target_centroids_list.extend(centroids_list_dict[c1])
        target_len_trn.append(len(uri_list_dict[c1]) - lens_dict[c1])
        target_len_val.append(lens_dict[c1])
        target_val_idxs = target_val_idxs + tuple(range(target_val_idxs_index, target_val_idxs_index + lens_dict[c1]))
        target_val_idxs_index += len(uri_list_dict[c1])

    target_head = heads_dict[target_datasets[0]]

    if training_type == "3c" or training_type == "4c" or training_type == "5c":
        ##### md_target_datasets_sep: all target_datasets on separate heads
        fnames_dict = [target_uri_list[i][:-4] for i in range(len(target_uri_list))]
        centroids_dict = [target_centroids_list[i][1:] for i in range(len(target_uri_list))]
        df = pd.DataFrame({'fnames': fnames_dict, 'centroids': centroids_dict}, columns=['fnames', 'centroids'])
        df.to_csv(path3 + "centroids_" + str(len(target_datasets)) + ".csv", index=False)
        val_idxs = target_val_idxs
        CENT_CSV_TARGET_DATASETS = Path(PATH2, source_list + "/centroids_" + str(len(target_datasets)) +".csv")
        md_target_datasets_sep = ImageClassifierData.from_csv(path=PATH, folder=IMAGES,
                                                              csv_fname=CENT_CSV_TARGET_DATASETS,
                                                              val_idxs=val_idxs, tfms=tfms, bs=bs, suffix='.tif',
                                                              continuous=True,
                                                              num_workers=16, len_list_trn=target_len_trn,
                                                              len_list_val=target_len_val)
        trn_ds2 = ConcatLblDataset_trn(md_target_datasets_sep.trn_ds)
        val_ds2 = ConcatLblDataset_val(md_target_datasets_sep.val_ds)
        md_target_datasets_sep.trn_dl.dataset = trn_ds2
        md_target_datasets_sep.val_dl.dataset = val_ds2

    elif training_type == "2b":
        ##### md_source_datasets_shared: all source_datasets on a shared head
        fnames_dict = [source_uri_list[i][:-4] for i in range(len(source_uri_list))]
        centroids_dict = [source_centroids_list[i][1:] for i in range(len(source_uri_list))]
        df = pd.DataFrame({'fnames': fnames_dict, 'centroids': centroids_dict}, columns=['fnames','centroids'])
        df.to_csv(path3+"centroids_"+str(len(source_datasets))+".csv", index=False)
        val_idxs = source_val_idxs
        CENT_CSV_SOURCE_DATASETS = Path(PATH2, source_list + "/centroids_"+str(len(source_datasets))+".csv")
        md_source_datasets_shared = ImageClassifierData.from_csv(path=PATH, folder=IMAGES, csv_fname=CENT_CSV_SOURCE_DATASETS,
                                                      val_idxs=val_idxs, tfms=tfms, bs=bs, suffix='.tif', continuous=True,
                                                      num_workers=16, len_list_trn=source_len_trn,
                                                      len_list_val=source_len_val)
        trn_ds2 = ConcatLblDataset2(md_source_datasets_shared.trn_ds, heads_dict["gen"])
        val_ds2 = ConcatLblDataset2(md_source_datasets_shared.val_ds, heads_dict["gen"])
        md_source_datasets_shared.trn_dl.dataset = trn_ds2
        md_source_datasets_shared.val_dl.dataset = val_ds2

    elif training_type == "3b" or training_type == "4b" or training_type == "5b":
        ##### md_source_datasets_sep: all source_datasets on separate heads
        fnames_dict = [source_uri_list[i][:-4] for i in range(len(source_uri_list))]
        centroids_dict = [source_centroids_list[i][1:] for i in range(len(source_uri_list))]
        df = pd.DataFrame({'fnames': fnames_dict, 'centroids': centroids_dict}, columns=['fnames','centroids'])
        df.to_csv(path3+"centroids_"+str(len(source_datasets))+".csv", index=False)
        val_idxs = source_val_idxs
        CENT_CSV_SOURCE_DATASETS = Path(PATH2, source_list + "/centroids_"+str(len(source_datasets))+".csv")
        md_source_datasets_sep = ImageClassifierData.from_csv(path=PATH, folder=IMAGES, csv_fname=CENT_CSV_SOURCE_DATASETS,
                                                      val_idxs=val_idxs, tfms=tfms, bs=bs, suffix='.tif', continuous=True,
                                                      num_workers=16, len_list_trn=source_len_trn,
                                                      len_list_val=source_len_val)
        trn_ds2 = ConcatLblDataset_trn(md_source_datasets_sep.trn_ds)
        val_ds2 = ConcatLblDataset_val(md_source_datasets_sep.val_ds)
        md_source_datasets_sep.trn_dl.dataset = trn_ds2
        md_source_datasets_sep.val_dl.dataset = val_ds2

    ######### Making anchor boxes
    anc_grid = int(sz / 16)
    k = 1

    anc_offset = 1/(anc_grid*2)
    anc_x = np.repeat(np.linspace(anc_offset, 1-anc_offset, anc_grid), anc_grid)
    anc_y = np.tile(np.linspace(anc_offset, 1-anc_offset, anc_grid), anc_grid)

    anc_ctrs = V(np.tile(np.stack([anc_x,anc_y], axis=1), (k,1)), requires_grad=False).float()
    anc_sizes = np.array([[1/anc_grid,1/anc_grid] for i in range(anc_grid*anc_grid)])
    # **local
    # anchors = V(np.concatenate([anc_ctrs.data, anc_sizes], axis=1), requires_grad=False).float()
    anchors = V(np.concatenate([anc_ctrs.data.cpu(), anc_sizes], axis=1), requires_grad=False).float()

    grid_sizes = V(np.array([1/anc_grid]), requires_grad=False).unsqueeze(1)

    def hw2corners(ctr, hw): return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)
    anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])

    # anchor_cnr

    class GroupNorm(nn.Module):
        def __init__(self, num_features, num_groups=32, eps=1e-5):
            super(GroupNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
            self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
            self.num_groups = num_groups
            self.eps = eps

        def forward(self, x):
            N,C,H,W = x.size()
            G = self.num_groups
            assert C % G == 0

            x = x.view(N,G,-1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x-mean) / (var+self.eps).sqrt()
            x = x.view(N,C,H,W)
            return x * self.weight + self.bias

    class StdConv(nn.Module):
        def __init__(self, nin, nout, stride=2, drop=0.1):
            super().__init__()
            self.conv = nn.Conv2d(nin, nout, 3, stride=stride, padding=1)
    #         self.bn = nn.BatchNorm2d(nout)
            self.gn = GroupNorm(nout)
            self.drop = nn.Dropout(drop)

    #     def forward(self, x): return self.drop(F.relu(self.conv(x)))
    #     def forward(self, x): return self.drop(self.bn(F.relu(self.conv(x))))
        def forward(self, x): return self.drop(F.relu(self.gn(self.conv(x))))

    def flatten_conv(x, k):
        bs, nf, gx, gy = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.view(bs, -1, nf // k)

    class OutConv(nn.Module):
        def __init__(self, k, nin, bias):
            super().__init__()
            self.k = k
            self.oconv1 = nn.Conv2d(nin, (2) * k, 3, padding=1)

            self.oconv2 = nn.Conv2d(nin, 2 * k, 3, padding=1)
            self.oconv1.bias.data.zero_().add_(bias)

        def forward(self, x):
            return [flatten_conv(self.oconv1(x), self.k),
                    flatten_conv(self.oconv2(x), self.k)]

    def one_hot_embedding(labels, num_classes):
        return torch.eye(num_classes)[labels.data.cpu()]

    class BCE_Loss(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.num_classes = num_classes

        def forward(self, pred, targ):
            t = one_hot_embedding(targ.type(torch.LongTensor), self.num_classes + 1)
            t = V(t[:, 1:].contiguous())  # .cpu()
            x = pred[:, 1:]
            w = self.get_weight(x, t)
            # **local
            # return F.binary_cross_entropy_with_logits(x, t, w, size_average=False) / self.num_classes
            return F.binary_cross_entropy_with_logits(x, t, reduction='sum') / self.num_classes

        def get_weight(self, x, t): return None

    class FocalLoss(BCE_Loss):
        def get_weight(self,x,t):
            alpha,gamma = 0.25,2.
            p = x.sigmoid()
            pt = p*t + (1-p)*(1-t)
            w = alpha*t + (1-alpha)*(1-t)
            return w * (1-pt).pow(gamma)

    loss_f = FocalLoss(1)

    auxilary.auxilary.Tparticle[0] = target_head*torch.eye(1, dtype=torch.int8)

    class SSD_Head(nn.Module):
        def __init__(self, k, bias, drop=0.3):
            super().__init__()

            self.aux = auxilary.auxilary()

            self.drop = []
            self.sconv0 = []
            self.sconv1 = []
            self.sconv2 = []
            self.sconv3 = []
            self.out = []
            for i in range(38 + 1):
                self.drop.append(nn.Dropout(drop))
                self.add_module('drop' + str(i), self.drop[i])
                self.sconv0.append(StdConv(64, 128, stride=2, drop=drop))
                self.add_module('sconv0' + str(i), self.sconv0[i])
                self.sconv1.append(StdConv(128, 256, stride=2, drop=drop))
                self.add_module('sconv1' + str(i), self.sconv1[i])
                self.sconv2.append(StdConv(256, 512, stride=2, drop=drop))
                self.add_module('sconv2' + str(i), self.sconv2[i])
                self.sconv3.append(StdConv(512, 1024, stride=2, drop=drop))
                self.add_module('sconv3' + str(i), self.sconv3[i])
                self.out.append(OutConv(k, 1024, bias))
                self.add_module('out' + str(i), self.out[i])

        def forward(self, x):
            self.Tparticle = self.aux.Tparticle[0]
            x = self.drop[self.Tparticle](F.relu(x))
            x = self.sconv0[self.Tparticle](x)
            x = self.sconv1[self.Tparticle](x)
            x = self.sconv2[self.Tparticle](x)
            x = self.sconv3[self.Tparticle](x)
            return self.out[self.Tparticle](x)

    head_reg4 = SSD_Head(k, -4.)
    models = ConvnetBuilder(f_model, 0, 0, 0, xtra_fc=[1024], custom_head=head_reg4, nfdef=128)

    def intersect(box_a, box_b):
        """ Returns the intersection of two boxes """
        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def box_sz(b):
        """ Returns the box size"""
        return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def jaccard(box_a, box_b):
        """ Returns the jaccard distance between two boxes"""
        inter = intersect(box_a, box_b)
        union = box_sz(box_a).unsqueeze(1) + box_sz(box_b).unsqueeze(0) - inter
        return inter / union

    def get_y(cent):
        """ gets rid of any of the bounding boxes that are just padding """
        cent = cent.float().view(-1,2)/sz
        # cent_keep = ((cent[:,0]+cent[:,1])>0).nonzero()[:,0]
        # resulting_cents = cent[cent_keep]
        resulting_cents = cent[(cent[:, 0] > 0) | (cent[:, 1] > 0)]
    #     return cent[cent_keep], clas[clas > 0]
        return resulting_cents, V(torch.ones(len(resulting_cents)))

    def actn_to_cent(actn, anc_ctrs):
        """ activations to centroids """
        actn_cents = torch.tanh(actn)
        # actn_bbs = torch.tanh(actn)
        # actn_centers = (actn_bbs[:, :2] / 2 * grid_sizes) + anchors[:, :2]
        # actn_hw = (actn_bbs[:, 2:] / 2 + 1) * anchors[:, 2:]
        return (actn_cents * 0.75 * grid_sizes) + anc_ctrs

    def map_to_ground_truth(overlaps):
        """ ... """
        prior_overlap, prior_idx = overlaps.max(1)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap,gt_idx

    def cent2bb(cent):
        bb = torch.zeros(len(cent), 4)
        par_szs = (torch.ones(len(cent)) * par_sz).cuda()
        torzeros = torch.zeros(len(cent)).cuda()
        torones = torch.ones(len(cent)).cuda()
        bb[:, 0] = torch.max(cent[:, 0] - par_szs / 2, torzeros)
        bb[:, 1] = torch.max(cent[:, 1] - par_szs / 2, torzeros)
        bb[:, 2] = torch.min(cent[:, 0] + par_szs / 2, torones)
        bb[:, 3] = torch.min(cent[:, 1] + par_szs / 2, torones)
        return bb.cuda()

    def ssd_1_loss(b_clas,b_cent,cent,clas):
        # cent,clas = get_y(cent,clas)
        cent, clas = get_y(cent)
        # a_ic = actn_to_bb(b_bb, anchors)
        b_cent = actn_to_cent(b_cent, anc_ctrs)
        bbox = cent2bb(cent.data)#.cpu())
        overlaps = jaccard(bbox, anchor_cnr.data)#.cpu())
        gt_overlap,gt_idx = map_to_ground_truth(overlaps)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > iou_trn
        pos_idx = torch.nonzero(pos)[:,0]
    #     gt_clas[1-pos] = 20
        gt_clas[1 - pos] = 0
        gt_cent = cent[gt_idx]
        loc_loss = ((b_cent[pos_idx] - gt_cent[pos_idx]).abs()).mean()
        clas_loss  = loss_f(b_clas, gt_clas)
        return loc_loss, clas_loss

    pdist = nn.PairwiseDistance(p=2, eps=1e-9)
    # def pdist(x1, x2): return torch.sqrt((torch.abs(x1 - x2) * torch.abs(x1 - x2)).sum())

    def lbias4(training_type, regularizer):
        if training_type == '4c':
            datasets = target_datasets
        elif training_type == '4b':
            datasets = source_datasets
        loss = 0
        modules = learn.get_layer_groups()[2]._modules['0']
        for c1 in datasets:
            for c2 in range(4):
                module1 = getattr(modules, "sconv" + str(c2) + str(heads_dict[c1]))
                module2 = getattr(modules, "sconv" + str(c2) + str(heads_dict[regularizer]))
                loss += pdist(module1.conv.weight.view(1,-1), module2.conv.weight.view(1,-1)).sum()
                loss += pdist(module1.conv.bias.view(1,-1), module2.conv.bias.view(1,-1)).sum()
                loss += pdist(module1.gn.weight.view(1,-1), module2.gn.weight.view(1,-1)).sum()
                loss += pdist(module1.gn.bias.view(1,-1), module2.gn.bias.view(1,-1)).sum()
            module1 = getattr(modules, "out" + str(heads_dict[c1]))
            module2 = getattr(modules, "out" + str(heads_dict[regularizer]))
            loss += pdist(module1.oconv1.weight.view(1,-1), module2.oconv1.weight.view(1,-1)).sum()
            loss += pdist(module1.oconv1.bias.view(1,-1), module2.oconv1.bias.view(1,-1)).sum()
            loss += pdist(module1.oconv2.weight.view(1,-1), module2.oconv2.weight.view(1,-1)).sum()
            loss += pdist(module1.oconv2.bias.view(1,-1), module2.oconv2.bias.view(1,-1)).sum()
        return loss

    def lbias5(training_type):
        if training_type == '5c':
            datasets = target_datasets
        elif training_type == '5b':
            datasets = source_datasets
        loss = 0
        modules = learn.get_layer_groups()[2]._modules['0']

        sconv_weight = []
        sconv_bias = []
        gn_weight = []
        gn_bias = []
        for i2 in range(4):
            sconv_weight.append(torch.mul(rgetattr(modules, "sconv" + str(i2) + str(9) + ".conv.weight"),0))
            sconv_bias.append(torch.mul(rgetattr(modules, "sconv" + str(i2) + str(9) + ".conv.bias"),0))
            gn_weight.append(torch.mul(rgetattr(modules, "sconv" + str(i2) + str(9) + ".gn.weight"),0))
            gn_bias.append(torch.mul(rgetattr(modules, "sconv" + str(i2) + str(9) + ".gn.bias"),0))

        oconv1_weight = torch.mul(rgetattr(modules, "out9" + ".oconv1.weight"), 0)
        oconv1_bias = torch.mul(rgetattr(modules, "out9" + ".oconv1.bias"), 0)
        oconv2_weight = torch.mul(rgetattr(modules, "out9" + ".oconv2.weight"), 0)
        oconv2_bias = torch.mul(rgetattr(modules, "out9" + ".oconv2.bias"), 0)

        for c1 in datasets:
            for i2 in range(4):
                sconv_weight[i2] = torch.add(sconv_weight[i2], rgetattr(modules, "sconv" + str(i2) + str(heads_dict[c1]) + ".conv.weight"))
                sconv_bias[i2] = torch.add(sconv_bias[i2], rgetattr(modules, "sconv" + str(i2) + str(heads_dict[c1]) + ".conv.bias"))
                gn_weight[i2] = torch.add(gn_weight[i2], rgetattr(modules, "sconv" + str(i2) + str(heads_dict[c1]) + ".gn.weight"))
                gn_bias[i2] = torch.add(gn_bias[i2], rgetattr(modules, "sconv" + str(i2) + str(heads_dict[c1]) + ".gn.bias"))
            oconv1_weight = torch.add(oconv1_weight, rgetattr(modules, "out" + str(heads_dict[c1]) + ".oconv1.weight"))
            oconv1_bias = torch.add(oconv1_bias, rgetattr(modules, "out" + str(heads_dict[c1]) + ".oconv1.bias"))
            oconv2_weight = torch.add(oconv2_weight, rgetattr(modules, "out" + str(heads_dict[c1]) + ".oconv2.weight"))
            oconv2_bias = torch.add(oconv2_bias, rgetattr(modules, "out" + str(heads_dict[c1]) + ".oconv2.bias"))

        datasets_count = len(datasets)
        for i2 in range(4):
            sconv_weight[i2] = torch.div(sconv_weight[i2], datasets_count)
            sconv_bias[i2] = torch.div(sconv_bias[i2], datasets_count)
            gn_weight[i2] = torch.div(gn_weight[i2], datasets_count)
            gn_bias[i2] = torch.div(gn_bias[i2], datasets_count)
        oconv1_weight = torch.div(oconv1_weight, datasets_count)
        oconv1_bias = torch.div(oconv1_bias, datasets_count)
        oconv2_weight = torch.div(oconv2_weight, datasets_count)
        oconv2_bias = torch.div(oconv2_bias, datasets_count)

        for c1 in datasets:
            for i2 in range(4):
                module1 = getattr(modules, "sconv" + str(i2) + str(heads_dict[c1]))
                loss += pdist(module1.conv.weight.view(1,-1), sconv_weight[i2].view(1,-1)).sum()
                loss += pdist(module1.conv.bias.view(1,-1), sconv_bias[i2].view(1,-1)).sum()
                loss += pdist(module1.gn.weight.view(1,-1), gn_weight[i2].view(1,-1)).sum()
                loss += pdist(module1.gn.bias.view(1,-1), gn_bias[i2].view(1,-1)).sum()
            module1 = getattr(modules, "out" + str(heads_dict[c1]))
            loss += pdist(module1.oconv1.weight.view(1,-1), oconv1_weight.view(1,-1)).sum()
            loss += pdist(module1.oconv1.bias.view(1,-1), oconv1_bias.view(1,-1)).sum()
            loss += pdist(module1.oconv2.weight.view(1,-1), oconv2_weight.view(1,-1)).sum()
            loss += pdist(module1.oconv2.bias.view(1,-1), oconv2_bias.view(1,-1)).sum()
        return loss

    def ssd_loss(pred,targ):
        # global unbiased_training
        nonlocal unbiased_training
        # global counter
        # counter += 1
        lcs,lls = 0.,0.
        for b_clas,b_cent,cent,clas,dsnum in zip(*pred,*targ):
            # print('dsnum: ', dsnum)
            loc_loss,clas_loss = ssd_1_loss(b_clas,b_cent,cent,clas)
            lls += loc_loss
            lcs += clas_loss
        if unbiased_training:
            if training_type == '4b' or training_type == '4c':
                if train_fine_tune_head == 1:
                    ratio1 = len(source_datasets) / (len(source_datasets) + len(target_datasets))
                    ratio2 = len(target_datasets) / (len(source_datasets) + len(target_datasets))
                    lbs = ratio1 * lbias4(training_type, "gen") + ratio2 * lbias4(training_type, "fine_tuned")
                else:
                    lbs = lbias4(training_type, "gen")
            elif training_type == '5b' or training_type == '5c':
                lbs = lbias5(training_type)
            else:
                print('Error: Wrong request of unbiased training!')
                exit(1)
        else:
            lbs = 0
        # 13500 is a multiplier to make lls and lcs approximately equal
        return 13500*lls+lcs+(gen_multiplier)*lbs

    ######### Making anchor boxes
    def make_ctrs_tst(sizes):
        anc_grid0 = int(sizes[0] / 16)
        anc_grid1 = int(sizes[1] / 16)
        k = 1

        anc_offset0 = 1 / (anc_grid0 * 2)
        anc_offset1 = 1 / (anc_grid1 * 2)
        # anc_x = np.repeat(np.linspace(anc_offset, 1-anc_offset, anc_grid), anc_grid)
        anc_x = np.repeat(np.linspace(anc_offset0, 1 - anc_offset0, anc_grid0), anc_grid1)
        # anc_y = np.tile(np.linspace(anc_offset, 1-anc_offset, anc_grid), anc_grid)
        anc_y = np.tile(np.linspace(anc_offset1, 1 - anc_offset1, anc_grid1), anc_grid0)

        anc_ctrs_tst = V(np.tile(np.stack([anc_x,anc_y], axis=1), (k,1)), requires_grad=False).float()
        # anc_sizes = np.array([[1/anc_grid,1/anc_grid] for i in range(anc_grid*anc_grid)])
        # anchors = V(np.concatenate([anc_ctrs.data, anc_sizes], axis=1), requires_grad=False).float()

        # grid_sizes = V(np.array([1/anc_grid0]), requires_grad=False).unsqueeze(1)
        grid_sizes_tst = V(np.array([1 / anc_grid0 , 1 / anc_grid1]), requires_grad=False).unsqueeze(0)
        return anc_ctrs_tst, grid_sizes_tst

    def actn_to_cent_tst(actn, sizes):
        """ activations to centroids """
        actn_cents = torch.tanh(actn)
        anc_ctrs_tst, grid_sizes_tst = make_ctrs_tst(sizes)
        return (actn_cents * 0.75 * grid_sizes_tst) + anc_ctrs_tst

    def get_y_tst(cent, sizes):
        """ gets rid of any of the bounding boxes that are just padding """
        cent = torch.cat([(cent.float().view(-1, 2)[:, 0] / sizes[0]).unsqueeze(1),
                   (cent.float().view(-1, 2)[:, 1] / sizes[1]).unsqueeze(1)], 1)
        # cent = cent.float().view(-1,2)/sz
        # cent_keep = ((cent[:,0]+cent[:,1])>0).nonzero()[:,0]
        # resulting_cents = cent[cent_keep]
        resulting_cents = cent[(cent[:, 0] > 0) | (cent[:, 1] > 0)]
    #     return cent[cent_keep], clas[clas > 0]
        return resulting_cents, V(torch.ones(len(resulting_cents)))

        TPcenters = np.zeros(pred[1].shape[1], dtype=float)
        if not par_size:
            par_size = np.zeros((2), dtype=np.float64)
            if five_crop:
                par_size[0] = par_sz
                par_size[1] = par_sz
            else:
                par_size[0] = par_sz * 368 / sizes[0]
                par_size[1] = par_sz * 368 / sizes[1]
        for b_clas, b_cent, cent, clas, dsnum in zip(*pred, *targ):


            if plotting:
                clas_ids = b_clas
            else:
                if five_crop:
                    anchszsq = int(pred[1].shape[1] / 4)
                    cent, clas = get_y(cent)

                    clas_pr, clas_ids = b_clas.max(1)
                    clas_pr = clas_pr.sigmoid().data.cpu().numpy()
                    clas_ids = clas_ids.data.cpu().numpy()

                    b_cent1 = actn_to_cent(b_cent[0:anchszsq], anc_ctrs)
                    b_cent1[:, 0] = b_cent1[:, 0] + ((sizes[0] - sz) / sz)
                    b_cent1[:, 1] = b_cent1[:, 1] + ((sizes[1] - sz) / sz)
                    clas_ids1 = clas_ids[0:anchszsq]
                    clas_ids1[((b_cent1[:, 0] < (((sizes[0]) / sz) / 2)).data.cpu().numpy() == 1) + (
                                (b_cent1[:, 1] < (((sizes[1]) / sz) / 2)).data.cpu().numpy() == 1)] = 0

                    b_cent2 = actn_to_cent(b_cent[anchszsq:anchszsq * 2], anc_ctrs)
                    b_cent2[:, 1] = b_cent2[:, 1] + ((sizes[1] - sz) / sz)
                    clas_ids2 = clas_ids[anchszsq:anchszsq * 2]
                    clas_ids2[((b_cent2[:, 0] >= (((sizes[0]) / sz) / 2)).data.cpu().numpy() == 1) + (
                            (b_cent2[:, 1] < (((sizes[1]) / sz) / 2)).data.cpu().numpy() == 1)] = 0

                    b_cent3 = actn_to_cent(b_cent[anchszsq * 2:anchszsq * 3], anc_ctrs)
                    b_cent3[:, 0] = b_cent3[:, 0] + ((sizes[0] - sz) / sz)
                    clas_ids3 = clas_ids[anchszsq * 2:anchszsq * 3]
                    clas_ids3[((b_cent3[:, 0] < (((sizes[0]) / sz) / 2)).data.cpu().numpy() == 1) + (
                            (b_cent3[:, 1] >= (((sizes[1]) / sz) / 2)).data.cpu().numpy() == 1)] = 0

                    b_cent4 = actn_to_cent(b_cent[anchszsq * 3:anchszsq * 4], anc_ctrs)
                    clas_ids4 = clas_ids[anchszsq * 3:anchszsq * 4]
                    clas_ids4[((b_cent4[:, 0] >= (((sizes[0]) / sz) / 2)).data.cpu().numpy() == 1) + (
                            (b_cent4[:, 1] >= (((sizes[1]) / sz) / 2)).data.cpu().numpy() == 1)] = 0

                    b_cent = torch.cat([b_cent1, b_cent2, b_cent3, b_cent4], dim=0)
                    clas_ids = np.concatenate([clas_ids1, clas_ids2, clas_ids3, clas_ids4])
                else:
                    cent, clas = get_y_tst(cent, sizes)

                    b_cent = actn_to_cent_tst(b_cent, sizes)
                    clas_pr, clas_ids = b_clas.max(1)
                    clas_pr = clas_pr.sigmoid().data.cpu().numpy()
                    clas_ids = clas_ids.data.cpu().numpy()

            coor_x = np.broadcast_to(cent[:, 0].data.cpu().numpy(), (len(b_cent), len(cent)))
            coor_y = np.broadcast_to(cent[:, 1].data.cpu().numpy(), (len(b_cent), len(cent)))

            coor_mx = (np.broadcast_to(b_cent[:, 0].data.cpu().numpy(), (len(cent), len(b_cent)))).T
            abs_x = np.fabs(np.subtract(coor_x, coor_mx))
            coor_my = (np.broadcast_to(b_cent[:, 1].data.cpu().numpy(), (len(cent), len(b_cent)))).T
            abs_y = np.fabs(np.subtract(coor_y, coor_my))

            intersection = ((par_size[0] - abs_x) >= 0) * (par_size[0] - abs_x) * ((par_size[1] - abs_y) >= 0) * (
                        par_size[1] - abs_y)
            union = 2 * par_size[0] * par_size[1]
            IOUs = intersection / (union - intersection)

            matched = np.zeros(len(cent), dtype=np.uint8)
            tpcounter = 0
            for k in np.argsort(clas_pr)[::-1]:
                if clas_ids[k] == 1 and clas_pr[k] > conf_thresh:
                    neighbour_index = 0
                    neighbour = np.ones((2, 10), dtype=np.float32) * -1
                    for i in range(len(cent)):
                        if matched[i] == 0 and IOUs[k, i] > IOU_thresh and clas[i] == 1:
                            neighbour[0, neighbour_index] = i
                            neighbour[1, neighbour_index] = IOUs[k, i]
                            neighbour_index += 1

                    if neighbour_index >= 1:
                        if neighbour_index > 1:
                            indexes = np.argsort(neighbour[1, :], axis=0)
                            neighbour[0, 0] = neighbour[0, indexes[-1]]
                        index = int(neighbour[0, 0])
                        matched[index] = 1
                        TP = 1
                        FP = 0
                    else:
                        TP = 0
                        FP = 1
                    detections.append([clas_pr[k], TP, FP, 0, 0, 0])
                    tpcounter += 1
                    # &&
                    if tpcenters:# and clas_pr[k] > 0.5:
                        TPcenters[k] = 1
            total_reference += len(cent)
            # total_reference_neg += len(b_cent) - len(cent)
            total_reference_neg += (sizes[0]//16)*(sizes[1]//16) - len(cent)
            if tpcenters:

                def draw_outline(o, lw):
                    o.set_path_effects([patheffects.Stroke(
                        linewidth=lw, foreground='black'), patheffects.Normal()])

                def draw_rect(ax, b, color='white', lw=2, outline=1):
                    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=lw))
                    draw_outline(patch, outline)

                def show_img(im, sizes, figsize=None, ax=None):
                    if not ax: fig, ax = plt.subplots(figsize=figsize)
                    ax.imshow(im, cmap='gray', interpolation='none', filternorm=False)
                    ax.set_xticks(np.linspace(0, sizes[1], 8))
                    ax.set_yticks(np.linspace(0, sizes[0], 8))
                    # ax.grid()
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])
                    return ax

                def cc_hw(a):
                    b0 = a[1] - (par_sz_pix // 2)
                    if b0 < 0:
                        b0 = 0
                    b1 = a[0] - (par_sz_pix // 2)
                    if b1 < 0:
                        b1 = 0
                    return np.array([b0, b1, par_sz_pix, par_sz_pix])

                def show_ground_truth(ax, im, bbox1, bbox2, sizes, clas1=None, clas2=None, prs=None, thresh=0.3,
                                      TPcenters=None):
                    bb1 = [cc_hw(o) for o in bbox1.reshape(-1, 2)]
                    bb2 = [cc_hw(o) for o in bbox2.reshape(-1, 2)]
                    if prs is None:
                        prs1 = [None] * len(bb1)
                        prs2 = [None] * len(bb2)
                    else:
                        prs1 = prs
                        prs2 = prs
                    if clas1 is None: clas1 = [None] * len(bb1)
                    if clas2 is None: clas2 = [None] * len(bb2)
                    TPcenters1 = [1] * len(bb1)
                    TPcenters2 = TPcenters
                    ax = show_img(im, sizes, ax=ax)
                    for i, (b, c, pr, tpc) in enumerate(zip(bb1, clas1, prs1, TPcenters1)):
                        if tpc > 0:# and tpc < 0.8:
                            draw_rect(ax, b, color=colr_list[int(c) + 1], lw=4)
                            # txt = str(round(tpc, 2))
                            # draw_text(ax, b[:2], txt, color=colr_list[c + 0])
                    for i, (b, c, pr, tpc) in enumerate(zip(bb2, clas2, prs2, TPcenters2)):
                        if tpc > 0:# and tpc < 0.8:
                            # draw_rect(ax, b, color=colr_list[int(c) + 0], lw=2, outline=2)
                            draw_rect(ax, b, color='red', lw=2, outline=2)
                            # txt = str(round(tpc, 2))
                            # draw_text(ax, b[:2], txt, color=colr_list[c + 0])

                def torch_gt(ax, ima, bbox1, clas1, bbox2, clas2, sizes, prs=None, thresh=0.4, TPcenters=None):
                    bbox1 = np.transpose(
                        np.array(to_np([(bbox1[:, 0] * sizes[0]).long(), (bbox1[:, 1] * sizes[1]).long()])))
                    bbox2 = np.transpose(
                        np.array(to_np([(bbox2[:, 0] * sizes[0]).long(), (bbox2[:, 1] * sizes[1]).long()])))
                    return show_ground_truth(ax, ima, bbox1, bbox2, sizes, to_np(clas1), to_np(clas2),
                                             to_np(prs) if prs is not None else None, thresh, TPcenters)

                def plot_results(thresh, x0, sizes, md, md_name, b_cent1=b_cent, clas_ids1=clas_ids, b_cent2=b_cent, clas_ids2=clas_ids, TPcenters=None):
                    x0 = to_np(x0)
                    fig, axes = plt.subplots(1, 1, figsize=(sizes[0]/25.0, sizes[1]/25.0))#figsize=(50.8, 49.2))
                    ima = md.val_ds.ds.denorm(x0)[0]
                    torch_gt(axes, ima.squeeze(), b_cent1, clas_ids1, b_cent2, clas_ids2, sizes, None,
                             clas_pr.max() * thresh, TPcenters)
                    plt.tight_layout()
                    fig.savefig("box_plot_"+ str(np.random.randint(10, 10000)) + md_name +".pdf")

                import matplotlib.cm as cmx
                import matplotlib.colors as mcolors

                def get_cmap(N):
                    color_norm = mcolors.Normalize(vmin=0, vmax=N - 1)
                    return cmx.ScalarMappable(norm=color_norm, cmap='Set3').to_rgba

                num_colr = 5
                cmap = get_cmap(num_colr)
                colr_list = [cmap(float(x)) for x in range(num_colr)]

                # plot_results(.5, x0, sizes, md, md_name, b_cent=cent, clas_ids=np.asarray(clas.cpu().numpy(), dtype=np.int))
                # plot_results(.5, x0, sizes, md, md_name,b_cent=b_cent, TPcenters=TPcenters)
                plot_results(.5, x0, sizes, md, md_name, b_cent1=cent,
                             clas_ids1=np.asarray(clas.cpu().numpy(), dtype=np.int), b_cent2=b_cent, TPcenters=TPcenters)

        return 1

    if training_type == "3c" or training_type == "4c" or training_type == "5c":
        learn = ConvLearner(md_target_datasets_sep, models)
    elif training_type == "2b":
        learn = ConvLearner(md_source_datasets_shared, models)
    elif training_type == "3b" or training_type == "4b" or training_type == "5b":
        learn = ConvLearner(md_source_datasets_sep, models)

    if optimizer_type == 'adam_sgdr' or optimizer_type == 'adam':
        learn.opt_fn = optim.Adam
    elif optimizer_type == 'sgd':
        learn.opt_fn = optim.SGD
    learn.crit = ssd_loss
    learn.unfreeze()
    learn.model.cuda()

    def copy_weights_2bto3b4b5b_or_4bto4c_or_5bto5c():
        copy_source = heads_dict["gen"]
        if training_type == '3b' or training_type == '4b' or training_type == '5b':
            datasets = source_datasets
        elif training_type == '4c' or training_type == '5c':
            datasets = target_datasets + ["fine_tuned"]
        for modules1 in [learn.models.model[1], learn.model[1], learn.models.fc_model[0]]:
            for c1 in datasets:
                for i2 in range(4):
                    rsetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".conv.weight.data", dc(rgetattr(modules1, "sconv" + str(i2) + str(copy_source) + ".conv.weight.data")))
                    rsetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".conv.bias.data", dc(rgetattr(modules1, "sconv" + str(i2) + str(copy_source) + ".conv.bias.data")))
                    rsetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".gn.weight.data", dc(rgetattr(modules1, "sconv" + str(i2) + str(copy_source) + ".gn.weight.data")))
                    rsetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".gn.bias.data", dc(rgetattr(modules1, "sconv" + str(i2) + str(copy_source) + ".gn.bias.data")))
                rsetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv1.weight.data", dc(rgetattr(modules1, "out" + str(copy_source) + ".oconv1.weight.data")))
                rsetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv1.bias.data", dc(rgetattr(modules1, "out" + str(copy_source) + ".oconv1.bias.data")))
                rsetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv2.weight.data", dc(rgetattr(modules1, "out" + str(copy_source) + ".oconv2.weight.data")))
                rsetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv2.bias.data", dc(rgetattr(modules1, "out" + str(copy_source) + ".oconv2.bias.data")))
        learn.model.zero_grad()
    # copy_weights_2bto3b4b5b_or_4bto4c_or_5bto5c()

    def copy_weights_3bto4b_or_3bto3c():
        datasets = source_datasets
        if training_type == '3c':
            targets = target_datasets
        elif training_type == '4b':
            targets = heads_dict["gen"]
        for modules1 in [learn.models.model[1], learn.model[1], learn.models.fc_model[0]]:
            sconv_weight = []
            sconv_bias = []
            gn_weight = []
            gn_bias = []
            for i2 in range(4):
                sconv_weight.append(torch.sub(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict["gen"]) + ".conv.weight.data"), dc(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict["gen"]) + ".conv.weight.data"))))
                sconv_bias.append(torch.sub(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict["gen"]) + ".conv.bias.data"), dc(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict["gen"]) + ".conv.bias.data"))))
                gn_weight.append(torch.sub(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict["gen"]) + ".gn.weight.data"), dc(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict["gen"]) + ".gn.weight.data"))))
                gn_bias.append(torch.sub(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict["gen"]) + ".gn.bias.data"), dc(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict["gen"]) + ".gn.bias.data"))))

            oconv1_weight = torch.sub(rgetattr(modules1, "out" + str(heads_dict["gen"]) + ".oconv1.weight.data"), dc(rgetattr(modules1, "out" + str(heads_dict["gen"]) + ".oconv1.weight.data")))
            oconv1_bias = torch.sub(rgetattr(modules1, "out" + str(heads_dict["gen"]) + ".oconv1.bias.data"), dc(rgetattr(modules1, "out" + str(heads_dict["gen"]) + ".oconv1.bias.data")))
            oconv2_weight = torch.sub(rgetattr(modules1, "out" + str(heads_dict["gen"]) + ".oconv2.weight.data"), dc(rgetattr(modules1, "out" + str(heads_dict["gen"]) + ".oconv2.weight.data")))
            oconv2_bias = torch.sub(rgetattr(modules1, "out" + str(heads_dict["gen"]) + ".oconv2.bias.data"), dc(rgetattr(modules1, "out" + str(heads_dict["gen"]) + ".oconv2.bias.data")))

            for c1 in datasets:
                for i2 in range(4):
                    sconv_weight[i2] = torch.add(sconv_weight[i2], dc(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".conv.weight.data")))
                    sconv_bias[i2] = torch.add(sconv_bias[i2], dc(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".conv.bias.data")))
                    gn_weight[i2] = torch.add(gn_weight[i2], dc(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".gn.weight.data")))
                    gn_bias[i2] = torch.add(gn_bias[i2], dc(rgetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".gn.bias.data")))
                oconv1_weight = torch.add(oconv1_weight, dc(rgetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv1.weight.data")))
                oconv1_bias = torch.add(oconv1_bias, dc(rgetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv1.bias.data")))
                oconv2_weight = torch.add(oconv2_weight, dc(rgetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv2.weight.data")))
                oconv2_bias = torch.add(oconv2_bias, dc(rgetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv2.bias.data")))

            datasets_count = len(datasets)
            for i2 in range(4):
                sconv_weight[i2] = torch.div(sconv_weight[i2], datasets_count)
                sconv_bias[i2] = torch.div(sconv_bias[i2], datasets_count)
                gn_weight[i2] = torch.div(gn_weight[i2], datasets_count)
                gn_bias[i2] = torch.div(gn_bias[i2], datasets_count)
            oconv1_weight = torch.div(oconv1_weight, datasets_count)
            oconv1_bias = torch.div(oconv1_bias, datasets_count)
            oconv2_weight = torch.div(oconv2_weight, datasets_count)
            oconv2_bias = torch.div(oconv2_bias, datasets_count)

            for c1 in targets:
                for i2 in range(4):
                    rsetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".conv.weight.data", sconv_weight[i2])
                    rsetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".conv.bias.data", sconv_bias[i2])
                    rsetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".gn.weight.data", gn_weight[i2])
                    rsetattr(modules1, "sconv" + str(i2) + str(heads_dict[c1]) + ".gn.bias.data", gn_bias[i2])
                rsetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv1.weight.data", oconv1_weight)
                rsetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv1.bias.data", oconv1_bias)
                rsetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv2.weight.data", oconv2_weight)
                rsetattr(modules1, "out" + str(heads_dict[c1]) + ".oconv2.bias.data", oconv2_bias)
        learn.model.zero_grad()
    # copy_weights_3bto4b_or_3bto3c()

    def set_weights_5():
        if training_type == '5c':
            datasets = source_datasets
            datasets.append("gen")
        elif training_type == '5b':
            datasets = source_datasets
        for modules in [learn.models.model[1], learn.model[1], learn.models.fc_model[0]]:
            sconv_weight = []
            sconv_bias = []
            gn_weight = []
            gn_bias = []
            for i2 in range(4):
                sconv_weight.append(torch.mul(dc(rgetattr(modules, "sconv" + str(i2) + str(heads_dict["gen"]) + ".conv.weight.data")),0))
                sconv_bias.append(torch.mul(dc(rgetattr(modules, "sconv" + str(i2) + str(heads_dict["gen"]) + ".conv.bias.data")),0))
                gn_weight.append(torch.mul(dc(rgetattr(modules, "sconv" + str(i2) + str(heads_dict["gen"]) + ".gn.weight.data")),0))
                gn_bias.append(torch.mul(dc(rgetattr(modules, "sconv" + str(i2) + str(heads_dict["gen"]) + ".gn.bias.data")),0))

            oconv1_weight = torch.mul(dc(rgetattr(modules, "out" + str(heads_dict["gen"]) + ".oconv1.weight.data")), 0)
            oconv1_bias = torch.mul(dc(rgetattr(modules, "out" + str(heads_dict["gen"]) + ".oconv1.bias.data")), 0)
            oconv2_weight = torch.mul(dc(rgetattr(modules, "out" + str(heads_dict["gen"]) + ".oconv2.weight.data")), 0)
            oconv2_bias = torch.mul(dc(rgetattr(modules, "out" + str(heads_dict["gen"]) + ".oconv2.bias.data")), 0)

            for c1 in datasets:
                for i2 in range(4):
                    sconv_weight[i2] = torch.add(sconv_weight[i2], dc(rgetattr(modules, "sconv" + str(i2) + str(heads_dict[c1]) + ".conv.weight.data")))
                    sconv_bias[i2] = torch.add(sconv_bias[i2], dc(rgetattr(modules, "sconv" + str(i2) + str(heads_dict[c1]) + ".conv.bias.data")))
                    gn_weight[i2] = torch.add(gn_weight[i2], dc(rgetattr(modules, "sconv" + str(i2) + str(heads_dict[c1]) + ".gn.weight.data")))
                    gn_bias[i2] = torch.add(gn_bias[i2], dc(rgetattr(modules, "sconv" + str(i2) + str(heads_dict[c1]) + ".gn.bias.data")))
                oconv1_weight = torch.add(oconv1_weight, dc(rgetattr(modules, "out" + str(heads_dict[c1]) + ".oconv1.weight.data")))
                oconv1_bias = torch.add(oconv1_bias, dc(rgetattr(modules, "out" + str(heads_dict[c1]) + ".oconv1.bias.data")))
                oconv2_weight = torch.add(oconv2_weight, dc(rgetattr(modules, "out" + str(heads_dict[c1]) + ".oconv2.weight.data")))
                oconv2_bias = torch.add(oconv2_bias, dc(rgetattr(modules, "out" + str(heads_dict[c1]) + ".oconv2.bias.data")))

            datasets_count = len(datasets)
            for i2 in range(4):
                sconv_weight[i2] = torch.div(sconv_weight[i2], datasets_count)
                sconv_bias[i2] = torch.div(sconv_bias[i2], datasets_count)
                gn_weight[i2] = torch.div(gn_weight[i2], datasets_count)
                gn_bias[i2] = torch.div(gn_bias[i2], datasets_count)
            oconv1_weight = torch.div(oconv1_weight, datasets_count)
            oconv1_bias = torch.div(oconv1_bias, datasets_count)
            oconv2_weight = torch.div(oconv2_weight, datasets_count)
            oconv2_bias = torch.div(oconv2_bias, datasets_count)

            for i2 in range(4):
                rsetattr(modules, "sconv" + str(i2) + str(heads_dict["gen"]) + ".conv.weight.data", sconv_weight[i2])
                rsetattr(modules, "sconv" + str(i2) + str(heads_dict["gen"]) + ".conv.bias.data", sconv_bias[i2])
                rsetattr(modules, "sconv" + str(i2) + str(heads_dict["gen"]) + ".gn.weight.data", gn_weight[i2])
                rsetattr(modules, "sconv" + str(i2) + str(heads_dict["gen"]) + ".gn.bias.data", gn_bias[i2])
            rsetattr(modules, "out" + str(heads_dict["gen"]) + ".oconv1.weight.data", oconv1_weight)
            rsetattr(modules, "out" + str(heads_dict["gen"]) + ".oconv1.bias.data", oconv1_bias)
            rsetattr(modules, "out" + str(heads_dict["gen"]) + ".oconv2.weight.data", oconv2_weight)
            rsetattr(modules, "out" + str(heads_dict["gen"]) + ".oconv2.bias.data", oconv2_bias)
    # set_weights_5()

    if load is not None:
        if (convert_from is not None) and (convert_from != training_type):
            learn.load('SSPicker_'+load_version+'_' + convert_from + '_' + load)
            if training_type == "3c" or training_type == "4c" or training_type == "5c":
                learn.set_data(md_target_datasets_sep)
            elif training_type == "2b":
                learn.set_data(md_source_datasets_shared)
            elif training_type == "3b" or training_type == "4b" or training_type == "5b":
                learn.set_data(md_source_datasets_sep)

            if (training_type == '3b' and convert_from == '2b') or (
                    training_type == '4b' and convert_from == '2b') or (
                    training_type == '5b' and convert_from == '2b') or (
                    training_type == '4c' and convert_from == '4b') or (training_type == '5c' and convert_from == '5b'):
                copy_weights_2bto3b4b5b_or_4bto4c_or_5bto5c()
            elif (training_type == '4b' and convert_from == '3b') or (training_type == '3c' and convert_from == '3b'):
                copy_weights_3bto4b_or_3bto3c()
            elif (training_type == '5b' and convert_from == '3b'):
                set_weights_5()
            else:
                print('Error: Convertion types are not supported.')
                print('Supported conversions:')
                print('2b -> 3b/4b/5b')
                print('3b -> 3c/4b/5b')
                print('4b -> 4c')
                print('5b -> 5c')
                exit(1)
        else:
            learn.load('SSPicker_'+load_version+'_' + training_type + '_' + load)
        # learn.unfreeze()
        if heads_only:
            learn.freeze()
            if training_type == '4c' and train_gen_head == 0:
                for m in [learn.models.model[1], learn.model[1], learn.models.fc_model[0]]:
                    for module in [m.drop[heads_dict["gen"]], m.sconv0[heads_dict["gen"]], m.sconv1[heads_dict["gen"]],
                                   m.sconv2[heads_dict["gen"]], m.sconv3[heads_dict["gen"]], m.out[heads_dict["gen"]]]:
                        module.trainable = False
        else:
            learn.unfreeze()

    #if mode == 'predict':

    if prediction_head not in heads_dict:
        print("Error: Prediction head undefined!")
        exit(1)

    def add_border(x, y):
        sizes = torch.tensor(x.shape[2:])
        rmdr = torch.fmod(16 - torch.fmod(sizes, 16), 16)
        if torch.sum(rmdr) > 0:

            left = rmdr[0] / 2
            # right = ((rmdr[0]) + 1) / 2
            up = rmdr[1] / 2
            # down = ((rmdr[1]) + 1) / 2

            sizes_padded = sizes + rmdr
            x_padded = torch.zeros((1, 1, sizes_padded[0], sizes_padded[1]))
            x_padded[0, 0, left:left + sizes[0], up:up + sizes[1]] = x[0][0]
            # x_padded = torch.unsqueeze(x_padded, 0)
            # x_padded = torch.unsqueeze(x_padded, 0)

            if y:
                left = left.cuda()
                up = up.cuda()
                for i in range(len(y[0][0])):
                    if np.mod(i, 2) == 0:
                        y[0][0][i] += left
                    else:
                        y[0][0][i] += up

            return x_padded, y
        return x, y

    ######## Augmentations
    tfms0 = tfms_from_model(f_model, sz, crop_type=CropType.IDENTITY, tfm_y=TfmType.COORD_CENTERS, pad_mode=cv2.BORDER_REFLECT_101)

    ######### splitting for each dataset for testing

    uri_list_tst_dict = {"10077": [uri_list[i] for i in rand_10077[:3]],
                         "10078": [uri_list[i] for i in rand_10078[:3]],
                         "10081": [uri_list[i] for i in rand_10081[:3]],
                         "pdb_6bqv": [uri_list[i] for i in rand_6bqv[:2]],
                         "ss_1": [uri_list[i] for i in rand_ss_1[:3]],
                         "pdb_6bhu": [uri_list[i] for i in rand_6bhu[:2]],
                         "pdb_6bcx": [uri_list[i] for i in rand_6bcx[:2]],
                         "pdb_6bcq": [uri_list[i] for i in rand_6bcq[:2]],
                         "pdb_6bco": [uri_list[i] for i in rand_6bco[:2]],
                         "pdb_6az1": [uri_list[i] for i in rand_6az1[:2]],
                         "pdb_5y6p": [uri_list[i] for i in rand_5y6p[:3]],
                         "pdb_5xwy": [uri_list[i] for i in rand_5xwy[:2]],
                         "pdb_5w3s": [uri_list[i] for i in rand_5w3s[:2]],
                         "pdb_5ngm": [uri_list[i] for i in rand_5ngm[:2]],
                         "pdb_5mmi": [uri_list[i] for i in rand_5mmi[:2]],
                         "pdb_5foj": [uri_list[i] for i in rand_5foj[:1]],
                         "pdb_4zor": [uri_list[i] for i in rand_4zor[:2]],
                         "pdb_3j9i": [uri_list[i] for i in rand_3j9i[:2]],
                         "pdb_2gtl": [uri_list[i] for i in rand_2gtl[:2]],
                         "pdb_1sa0": [uri_list[i] for i in rand_1sa0[:2]],
                         "lf_1": [uri_list[i] for i in rand_lf_1[:3]],
                         "hh_2": [uri_list[i] for i in rand_hh_2[:3]],
                         "gk_1": [uri_list[i] for i in rand_gk_1[:3]],
                         "10156": [uri_list[i] for i in rand_10156[:3]],
                         "10153": [uri_list[i] for i in rand_10153[:3]],
                         "10122": [uri_list[i] for i in rand_10122[:3]],
                         "10097": [uri_list[i] for i in rand_10097[:3]],
                         "10089": [uri_list[i] for i in rand_10089[:3]],
                         "10084": [uri_list[i] for i in rand_10084[:3]],
                         "10017": [uri_list[i] for i in rand_10017[:3]],
                         "pdb_6b7n": [uri_list[i] for i in rand_6b7n[:2]],
                         "pdb_6b44": [uri_list[i] for i in rand_6b44[:2]],
                         "pdb_5xnl": [uri_list[i] for i in rand_5xnl[:2]],
                         "pdb_5w3l": [uri_list[i] for i in rand_5w3l[:2]],
                         "pdb_5vy5": [uri_list[i] for i in rand_5vy5[:1]],
                         "pdb_4hhb": [uri_list[i] for i in rand_4hhb[:1]],
                         "pdb_2wri": [uri_list[i] for i in rand_2wri[:2]]}

    centroids_list_tst_dict = {"10077": [centroids_list[i] for i in rand_10077[:3]],
                               "10078": [centroids_list[i] for i in rand_10078[:3]],
                               "10081": [centroids_list[i] for i in rand_10081[:3]],
                               "pdb_6bqv": [centroids_list[i] for i in rand_6bqv[:2]],
                               "ss_1": [centroids_list[i] for i in rand_ss_1[:3]],
                               "pdb_6bhu": [centroids_list[i] for i in rand_6bhu[:2]],
                               "pdb_6bcx": [centroids_list[i] for i in rand_6bcx[:2]],
                               "pdb_6bcq": [centroids_list[i] for i in rand_6bcq[:2]],
                               "pdb_6bco": [centroids_list[i] for i in rand_6bco[:2]],
                               "pdb_6az1": [centroids_list[i] for i in rand_6az1[:2]],
                               "pdb_5y6p": [centroids_list[i] for i in rand_5y6p[:3]],
                               "pdb_5xwy": [centroids_list[i] for i in rand_5xwy[:2]],
                               "pdb_5w3s": [centroids_list[i] for i in rand_5w3s[:2]],
                               "pdb_5ngm": [centroids_list[i] for i in rand_5ngm[:2]],
                               "pdb_5mmi": [centroids_list[i] for i in rand_5mmi[:2]],
                               "pdb_5foj": [centroids_list[i] for i in rand_5foj[:1]],
                               "pdb_4zor": [centroids_list[i] for i in rand_4zor[:2]],
                               "pdb_3j9i": [centroids_list[i] for i in rand_3j9i[:2]],
                               "pdb_2gtl": [centroids_list[i] for i in rand_2gtl[:2]],
                               "pdb_1sa0": [centroids_list[i] for i in rand_1sa0[:2]],
                               "lf_1": [centroids_list[i] for i in rand_lf_1[:3]],
                               "hh_2": [centroids_list[i] for i in rand_hh_2[:3]],
                               "gk_1": [centroids_list[i] for i in rand_gk_1[:3]],
                               "10156": [centroids_list[i] for i in rand_10156[:3]],
                               "10153": [centroids_list[i] for i in rand_10153[:3]],
                               "10122": [centroids_list[i] for i in rand_10122[:3]],
                               "10097": [centroids_list[i] for i in rand_10097[:3]],
                               "10089": [centroids_list[i] for i in rand_10089[:3]],
                               "10084": [centroids_list[i] for i in rand_10084[:3]],
                               "10017": [centroids_list[i] for i in rand_10017[:3]],
                               "pdb_6b7n": [centroids_list[i] for i in rand_6b7n[:2]],
                               "pdb_6b44": [centroids_list[i] for i in rand_6b44[:2]],
                               "pdb_5xnl": [centroids_list[i] for i in rand_5xnl[:2]],
                               "pdb_5w3l": [centroids_list[i] for i in rand_5w3l[:2]],
                               "pdb_5vy5": [centroids_list[i] for i in rand_5vy5[:1]],
                               "pdb_4hhb": [centroids_list[i] for i in rand_4hhb[:1]],
                               "pdb_2wri": [centroids_list[i] for i in rand_2wri[:2]]}

    source_uri_list_tst = []
    source_centroids_list_tst = []
    source_tst_idxs = ()
    source_tst_idxs_index = 0
    for c1 in source_datasets:
        source_uri_list_tst.extend(uri_list_tst_dict[c1])
        source_centroids_list_tst.extend(centroids_list_tst_dict[c1])
        source_tst_idxs = source_tst_idxs + tuple(
            range(source_tst_idxs_index, source_tst_idxs_index + lens_dict[c1]))
        source_tst_idxs_index += len(uri_list_tst_dict[c1])

    target_uri_list_tst = []
    target_centroids_list_tst = []
    target_tst_idxs = ()
    target_tst_idxs_index = 0
    for c1 in target_datasets:
        target_uri_list_tst.extend(uri_list_tst_dict[c1])
        target_centroids_list_tst.extend(centroids_list_tst_dict[c1])
        target_tst_idxs = target_tst_idxs + tuple(
            range(target_tst_idxs_index, target_tst_idxs_index + lens_dict[c1]))
        target_tst_idxs_index += len(uri_list_tst_dict[c1])

    if training_type == "2b" or training_type == "4b" or training_type == "5b" or training_type == "4c" or training_type == "5c":
        fnames_dict = [target_uri_list_tst[i][:-4] for i in range(len(target_uri_list_tst))]
        centroids_dict = [target_centroids_list_tst[i][1:] for i in range(len(target_uri_list_tst))]
        df = pd.DataFrame({'fnames': fnames_dict, 'centroids': centroids_dict}, columns=['fnames', 'centroids'])
        df.to_csv(path3 + "centroids_" + str(len(target_datasets)) + ".csv", index=False)
        val_idxs = target_tst_idxs
        CENT_CSV_TARGET_DATASETS = Path(PATH2, source_list + "/centroids_" + str(len(target_datasets)) + ".csv")

        auxilary.auxilary.Tparticle[0] = heads_dict[prediction_head] * torch.eye(1, dtype=torch.int8)

        md_target_datasets_shared_tst0 = ImageClassifierData.from_csv(path=PATH, folder=IMAGES,
                                                                      csv_fname=CENT_CSV_TARGET_DATASETS,
                                                                      val_idxs=val_idxs, tfms=tfms0, bs=1,
                                                                      suffix='.tif', continuous=True,
                                                                      num_workers=16)
        val_ds2 = ConcatLblDataset2(md_target_datasets_shared_tst0.val_ds, heads_dict["gen"])
        md_target_datasets_shared_tst0.val_dl.dataset = val_ds2

        iter0 = iter(md_target_datasets_shared_tst0.val_dl)

        learn.model.eval()
        auxilary.auxilary.heads_eval_mode(learn.model[1])
        start_time = time.time()

        with torch.no_grad():
            for val_counter in range(len(val_idxs)):
                learn.set_data(md_target_datasets_shared_tst0)
                x0, y0 = next(iter0)
                x0, y0 = add_border(x0, y0)
                x0 = V(x0)
                pred0 = learn.model(x0)

                for b_clas, b_cent in zip(*pred0):
                    b_cent = actn_to_cent_tst(b_cent, x0.shape[2:])
                    b_cent = b_cent.data.cpu().numpy()
                    b_cent[:, 0] = b_cent[:, 0] * x0.shape[2]
                    b_cent[:, 1] = b_cent[:, 1] * x0.shape[3]
                    # b_cent = np.asarray(np.round(b_cent), np.int32)
                    clas_pr, clas_ids = b_clas.max(1)
                    clas_pr = clas_pr.sigmoid().data.cpu().numpy()
                    clas_ids = clas_ids.data.cpu().numpy()

                    predicted_centroids = ''
                    predicted_confs = ''
                    for k in range(len(clas_pr[::-1])):
                        if clas_ids[k] == 1 and clas_pr[k] > prediction_conf:
                            predicted_centroids = ','.join(
                                [predicted_centroids, str(b_cent[k][0]), str(b_cent[k][1])])
                            predicted_confs = ','.join([predicted_confs, str(clas_pr[k])])
                    predicted_centroids = predicted_centroids[1:]
                    predicted_confs = predicted_confs[1:]

                    if prediction_subfolder:
                        with open("data/boxnet/predictions/"+prediction_subfolder+'/'+fnames_dict[val_counter]+"_coords.txt", "w") as text_file:
                            print(predicted_centroids, file=text_file)
                        with open("data/boxnet/predictions/"+prediction_subfolder+'/'+fnames_dict[val_counter]+"_confs.txt", "w") as text_file:
                            print(predicted_confs, file=text_file)
                    else:
                        # pd.DataFrame(np_array).to_csv("path/to/file.csv")
                        with open("data/boxnet/predictions/"+fnames_dict[val_counter]+"_coords.txt", "w") as text_file:
                            print(predicted_centroids, file=text_file)
                        with open("data/boxnet/predictions/"+fnames_dict[val_counter]+"_confs.txt", "w") as text_file:
                            print(predicted_confs, file=text_file)

                    # with open("data/boxnet/predictions/"+fnames_dict[val_counter]+".txt", 'r') as text_file:
                    #     x = text_file.read().split(',')

        print('prediction_time: ', time.time() - start_time)

def main(argv=None):
    start()
    exit(0)


if __name__ == '__main__':
    main()