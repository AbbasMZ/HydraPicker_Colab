import sys, os
import pickle
import imageio
import numpy as np
from skimage import measure
from pathlib import Path
import json
from matplotlib import patches, patheffects
import time
from copy import deepcopy as dc
import argparse
import functools
import inspect

from fastai.conv_learner import *
from fastai.dataset import *

from fastai.resnet_29_6 import resnet3 as myresnet

def start():
    # check to make sure you set the device
    # cuda_id = 0
    # torch.cuda.set_device(cuda_id)

    parser = argparse.ArgumentParser(description='A cross dataset generalization study using 37 Cryo-EM datasets.')
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['lrfind', 'train'], dest='mode')
    parser.add_argument('-t', '--training_type', required=True, type=str,
                        choices=['2b', '3b', '4b', '5b', '3c', '4c', '5c'], dest='training_type')
    parser.add_argument('-o', '--optimizer_type', default='adam_sgdr', type=str,
                        choices=['adam_sgdr', 'adam', 'sgd'], dest='optimizer_type')
    parser.add_argument('-e', '--epochs', default=250, type=int, dest='epochs')
    parser.add_argument('-c', '--cycle_len', default=40, type=int, dest='cycle_len')
    parser.add_argument('-g', '--gen_multiplier', type=float, default=0, dest='gen_multiplier')
    parser.add_argument('-s', '--save', type=str, default='0', dest='save')
    parser.add_argument('-l', '--load', type=str, default=None, dest='load')
    parser.add_argument('-cf', '--convert_from', type=str,
                        choices=['2b', '3b', '4b', '5b', '3c', '4c', '5c'], default=None, dest='convert_from')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, dest='lr')
    parser.add_argument('-wd', '--weight_decay', type=float, default=None, dest='wd')
    parser.add_argument('-uwds', '--use_weight_decay_scheduler', type=bool, default=False, dest='uwds')
    parser.add_argument('-lr0', '--learning_rate_0', type=float, default=None, dest='lr0')
    parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=1e-3, dest='lr_decay')
    parser.add_argument('-bs', '--batch_size', type=int, default=1, dest='bs')
    parser.add_argument('-sd', '--source_datasets',
                        default='10077$10078$10081$pdb_6bqv$ss_1$pdb_6bhu$pdb_6bcx$pdb_6bcq$pdb_6bco$pdb_6az1$pdb_5y6p$pdb_5xwy$pdb_5w3s$pdb_5ngm$pdb_5mmi$pdb_5foj$pdb_4zor$pdb_3j9i$pdb_2gtl$pdb_1sa0$lf_1$hh_2$gk_1$10156$10153$10122$10097$10089$10084$10017',
                        type=str, dest='source_dataset')
    parser.add_argument('-td', '--target_datasets',
                        default='pdb_6b7n$pdb_6b44$pdb_5xnl$pdb_5w3l$pdb_5vy5$pdb_4hhb$pdb_2wri', type=str,
                        dest='target_dataset')
    parser.add_argument('-cs', '--crop_size', type=int, default=368, dest='crop_size')  # up to 240 for original images
    parser.add_argument('-ps', '--particle_size', type=int, default=1, dest='particle_size')
    parser.add_argument('-si', '--source_image', type=str, default='micrographs', dest='source_image')
    parser.add_argument('-sl', '--source_list', type=str, default='labels', dest='source_list')
    parser.add_argument('-ioutrn', '--IOU_training', type=float, default=0.6, dest='iou_trn')
    parser.add_argument('-ioutst', '--IOU_testing', type=float, default=0.6, dest='iou_tst')
    parser.add_argument('-sn', '--serial_number', default='00000', type=str, dest='serial_number')
    parser.add_argument('-ho', '--heads_only', type=int, default=0, choices=[0, 1], dest='heads_only')
    parser.add_argument('-tgh', '--train_gen_head', type=int, default=1, choices=[0, 1], dest='train_gen_head')
    parser.add_argument('-tfth', '--train_fine_tune_head', type=int, default=0, choices=[0, 1], dest='train_fine_tune_head')
    parser.add_argument('-cp', '--check_pointing', type=int, default=1, choices=[0, 1], dest='check_pointing')
    args = parser.parse_args()

    def write_txt(fname, text_comments):
        with open(fname, 'wt') as f:
            f.write(text_comments + "\n\n")

    ############ Mainly used variables
    parameters_text = ''

    mode = args.mode
    parameters_text += "mode: " + mode

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

    serial_number = args.serial_number

    f_model = myresnet
    sz = args.crop_size
    parameters_text += "\ncrop_size: " + str(sz)

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
    # aux.aux.heads_training_only[0] = heads_only * torch.eye(1, dtype=torch.int8)
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
        # aux.aux.fine_tune[0] = 1 * torch.eye(1, dtype=torch.int8)
        aux.aux.fine_tune[0] = len(target_datasets) * torch.eye(1, dtype=torch.int8)
        for i in range(1, len(target_datasets) + 1):
            aux.aux.fine_tune[i] = heads_dict[target_datasets[i - 1]] * torch.eye(1, dtype=torch.int8)
        aux.aux.fine_tune_wgen[0] = train_gen_head * torch.eye(1, dtype=torch.int8)
        aux.aux.fine_tune_wgen[1] = heads_dict["gen"] * torch.eye(1, dtype=torch.int8)
        aux.aux.fine_tune_wfine[0] = train_fine_tune_head * torch.eye(1, dtype=torch.int8)
        aux.aux.fine_tune_wfine[1] = heads_dict["fine_tuned"] * torch.eye(1, dtype=torch.int8)
    else:
        aux.aux.fine_tune[0] = 0 * torch.eye(1, dtype=torch.int8)

    ############ Useful paths
    PATH = Path("data")
    PATH2 = Path("data")
    path3 = "data/" + source_list + "/"
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

    splits_dict = dict()
    lens_dict_tst = dict()
    lens_dict_vld = dict()
    uri_list_dict = dict()
    centroids_list_dict = dict()
    try:
        with open("data/splits.txt", "r") as text_file:
            splitsLines = text_file.readlines()

        for i in range(1, len(splitsLines)):
            splitsLine = splitsLines[i].split(', ')
            name = splitsLine[0]
            test = int(splitsLine[1])
            validation = int(splitsLine[2])
            lens_dict_tst[name] = int(test)
            lens_dict_vld[name] = int(validation)
            splits_dict[name] = list(map(int, splitsLine[3:]))

            uri_list_dict[name] = [uri_list[i] for i in splits_dict[name][lens_dict_tst[name]:]]
            centroids_list_dict[name] = [centroids_list[i] for i in splits_dict[name][lens_dict_tst[name]:]]

    except:
        print("Error reading splits file!")
        exit(1)

    # #### Number of all used datasets
    def numlist(count, number): return [number] * count

    num_list_trn = []
    num_list_val = []
    if training_type == "3b" or training_type == "4b" or training_type == "5b":
        for c1 in source_datasets:
            num_list_trn += numlist(len(uri_list_dict[c1]) - lens_dict_vld[c1], heads_dict[c1])
            num_list_val += numlist(lens_dict_vld[c1], heads_dict[c1])
    elif training_type == "3c" or training_type == "4c" or training_type == "5c":
        for c1 in target_datasets:
            num_list_trn += numlist(len(uri_list_dict[c1]) - lens_dict_vld[c1], heads_dict[c1])
            num_list_val += numlist(lens_dict_vld[c1], heads_dict[c1])

    class ConcatLblDataset_trn(Dataset):
        def __init__(self, ds):
            self.ds = ds
            self.sz = ds.sz

        def __len__(self): return len(self.ds)

        def __getitem__(self, i):
            x, y = self.ds[i]
            nonzeros = sum(np.sum(y.reshape(-1,2), 1) > 0)
            return (x, (y, np.ones(nonzeros), num_list_trn[i]))


    class ConcatLblDataset_val(Dataset):
        def __init__(self, ds):
            self.ds = ds
            self.sz = ds.sz

        def __len__(self): return len(self.ds)

        def __getitem__(self, i):
            x, y = self.ds[i]
            nonzeros = sum(np.sum(y.reshape(-1,2), 1) > 0)
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
            return (x, (y, np.ones(nonzeros), self.num))

    ######## Augmentations
    aug_tfms = [RandomRotate(180, mode=cv2.BORDER_REFLECT_101, tfm_y=TfmType.COORD_CENTERS),
                RandomFlip(tfm_y=TfmType.COORD_CENTERS)]
    tfms = tfms_from_model(f_model, sz, crop_type=CropType.LTDCENTER, tfm_y=TfmType.COORD_CENTERS, aug_tfms=aug_tfms,
                           pad_mode=cv2.BORDER_REFLECT_101)

    source_uri_list = []
    source_centroids_list = []
    source_len_trn = []
    source_len_val = []
    source_val_idxs = ()
    source_val_idxs_index = 0
    for c1 in source_datasets:
        source_uri_list.extend(uri_list_dict[c1])
        source_centroids_list.extend(centroids_list_dict[c1])
        source_len_trn.append(len(uri_list_dict[c1]) - lens_dict_vld[c1])
        source_len_val.append(lens_dict_vld[c1])
        source_val_idxs = source_val_idxs + tuple(range(source_val_idxs_index, source_val_idxs_index + lens_dict_vld[c1]))
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
        target_len_trn.append(len(uri_list_dict[c1]) - lens_dict_vld[c1])
        target_len_val.append(lens_dict_vld[c1])
        target_val_idxs = target_val_idxs + tuple(range(target_val_idxs_index, target_val_idxs_index + lens_dict_vld[c1]))
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
    anchors = V(np.concatenate([anc_ctrs.data, anc_sizes], axis=1), requires_grad=False).float()
    # anchors = V(np.concatenate([anc_ctrs.data.cpu(), anc_sizes], axis=1), requires_grad=False).float()

    grid_sizes = V(np.array([1/anc_grid]), requires_grad=False).unsqueeze(1)

    def hw2corners(ctr, hw): return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)
    anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])

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
            self.gn = GroupNorm(nout)
            self.drop = nn.Dropout(drop)

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
            t = V(t[:, 1:].contiguous())
            x = pred[:, 1:]
            w = self.get_weight(x, t)
            # **local
            return F.binary_cross_entropy_with_logits(x, t, w, size_average=False) / self.num_classes
            # return F.binary_cross_entropy_with_logits(x, t, reduction='sum') / self.num_classes

        def get_weight(self, x, t): return None

    class FocalLoss(BCE_Loss):
        def get_weight(self,x,t):
            alpha,gamma = 0.25,2.
            p = x.sigmoid()
            pt = p*t + (1-p)*(1-t)
            w = alpha*t + (1-alpha)*(1-t)
            return w * (1-pt).pow(gamma)

    loss_f = FocalLoss(1)

    aux.aux.Tparticle[0] = target_head*torch.eye(1, dtype=torch.int8)

    class SSD_Head(nn.Module):
        def __init__(self, k, bias, drop=0.3):
            super().__init__()

            self.aux = aux.aux()

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
        resulting_cents = cent[(cent[:, 0] > 0) | (cent[:, 1] > 0)]
        return resulting_cents, V(torch.ones(len(resulting_cents)))

    def actn_to_cent(actn, anc_ctrs):
        """ activations to centroids """
        actn_cents = torch.tanh(actn)
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
        cent, clas = get_y(cent)
        b_cent = actn_to_cent(b_cent, anc_ctrs)
        bbox = cent2bb(cent.data)
        overlaps = jaccard(bbox, anchor_cnr.data)
        gt_overlap,gt_idx = map_to_ground_truth(overlaps)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > iou_trn
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[1 - pos] = 0
        gt_cent = cent[gt_idx]
        loc_loss = ((b_cent[pos_idx] - gt_cent[pos_idx]).abs()).mean()
        clas_loss  = loss_f(b_clas, gt_clas)
        return loc_loss, clas_loss

    pdist = nn.PairwiseDistance(p=2, eps=1e-9)

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
        nonlocal unbiased_training
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

    if load is not None:
        if (convert_from is not None) and (convert_from != training_type):
            learn.load(load)
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
            learn.load(load)
        if heads_only:
            learn.freeze()
            if training_type == '4c' and train_gen_head == 0:
                for m in [learn.models.model[1], learn.model[1], learn.models.fc_model[0]]:
                    for module in [m.drop[heads_dict["gen"]], m.sconv0[heads_dict["gen"]], m.sconv1[heads_dict["gen"]],
                                   m.sconv2[heads_dict["gen"]], m.sconv3[heads_dict["gen"]], m.out[heads_dict["gen"]]]:
                        module.trainable = False
        else:
            learn.unfreeze()

    if mode == 'lrfind':

        lr = 5e-4
        if training_type == "3c" or training_type == "4c" or training_type == "5c":
            counts = (len(target_uri_list) - len(target_val_idxs))
            num_it = int(np.ceil(200.0 / counts)) * counts - 1
        elif training_type == "2b" or training_type == "3b" or training_type == "4b" or training_type == "5b":
            counts = (len(source_uri_list) - len(source_val_idxs))
            num_it = int(np.ceil(200.0 / counts)) * counts - 1

        if training_type == "2b":
            learn.lr_find2(start_lr=lr/1000, end_lr=0.05, num_it=num_it)
        elif training_type == "3b" or training_type == "4b" or training_type == "5b" or training_type == "3c" or training_type == "4c" or training_type == "5c":
                learn.lr_find2(start_lr=lr/1000, end_lr=0.05, num_it=num_it, particle=True)

        learn.save(save)

        learn.sched.plot(n_skip=1, n_skip_end=1)

        redo = input('Plot again [y/n] ? ')
        while redo == 'y':
            n_skip = int(input('Start: '))
            n_skip_end = int(input('End: '))
            learn.sched.plot(n_skip=n_skip, n_skip_end=n_skip_end)
            redo = input('Plot again [y/n] ? ')

    elif mode == 'train':

        write_txt("data/boxnet/params/" + serial_number + ".txt", parameters_text)
        if optimizer_type == 'adam':
            phase1 = TrainingPhase(epochs=epochs, opt_fn=optim.Adam, lr=lr, momentum=0.98)
            learn.fit_opt_sched([phase1], particle=True,
                                best_save_name=save + '_best',
                                save_path="data/boxnet/curves/" + save + '_')
        elif optimizer_type == 'sgd':
            phase1 = TrainingPhase(epochs=epochs, opt_fn=optim.SGD, lr=(lr, lr * lr_decay),
                                   lr_decay=DecayType.EXPONENTIAL, momentum=0.98)
            learn.fit_opt_sched([phase1], particle=True,
                                best_save_name=save + '_best',
                                save_path="data/boxnet/curves/" + save + '_')
        elif optimizer_type == 'adam_sgdr':
            if check_pointing:
                cycle_save_name = save
            else:
                cycle_save_name = None
            learn.fit(lr, epochs, cycle_len=cycle_len, particle=True, use_wd_sched=uwds, wds=wd,
                      best_save_name=save + '_best',
                      cycle_save_name=cycle_save_name,
                      save_path="data/boxnet/curves/" + save + '_')

        if training_type == '5b' or training_type == '5c':
            set_weights_5()

        learn.save(save)
        learn.sched.plot_loss(n_skip=0, n_skip_end=0, save_path="data/boxnet/curves/"+save+'_')

def main(argv=None):
    start()
    exit(0)

if __name__ == '__main__':
    main()