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

    parser = argparse.ArgumentParser(description='A cross dataset generalization study using 37 Cryo-EM datasets.')
    parser.add_argument('-m', '--mode', default='score', type=str, choices=['score', 'image'], dest='mode')
    parser.add_argument('-g', '--gen_multiplier', type=float, default=0, dest='gen_multiplier')
    parser.add_argument('-l', '--load', type=str, default=None, dest='load')
    parser.add_argument('-td', '--target_datasets',
                        default='pdb_6b7n$pdb_6b44$pdb_5xnl$pdb_5w3l$pdb_5vy5$pdb_4hhb$pdb_2wri', type=str,
                        dest='target_dataset')
    parser.add_argument('-cs', '--crop_size', type=int, default=368, dest='crop_size')  # up to 240 for original images
    parser.add_argument('-ps', '--particle_size', type=int, default=1, dest='particle_size')
    parser.add_argument('-si', '--source_image', type=str, default='micrographs', dest='source_image')
    parser.add_argument('-sl', '--source_list', type=str, default='labels', dest='source_list')
    parser.add_argument('-ioutst', '--IOU_testing', type=float, default=0.6, dest='iou_tst')
    parser.add_argument('-ph', '--prediction_head', type=str, default='gen', dest='prediction_head')
    parser.add_argument('-psf', '--prediction_subfolder', type=str, default=None, dest='prediction_subfolder')
    parser.add_argument('-pc', '--prediction_conf', type=float, default=0, dest='prediction_conf')
    parser.add_argument('-ef', '--evaluate_format', type=str, choices=['star', 'star_cryolo', 'star_boxnet'],
                        default='star', dest='evaluate_format')
    parser.add_argument('-cm', '--cryolo_model', type=str, default='phosnet', dest='cryolo_model')
    parser.add_argument('-bns', '--boxnet_suffix', type=str, default='_BoxNet_blank_trn5.star', dest='boxnet_suffix')
    parser.add_argument('-in', '--image_name', type=str, default='_image', dest='image_name')
    args = parser.parse_args()

    ############ Mainly used variables
    mode = args.mode
    if mode == 'image':
        draw = 1
    else:
        draw = 0
    load = args.load
    source_image = args.source_image
    source_list = args.source_list
    target_datasets = args.target_dataset.split('$')
    iou_tst = args.iou_tst
    f_model = myresnet
    sz = args.crop_size
    if args.particle_size == 0:
        par_sz_pix = 20
    elif args.particle_size == 1:
        par_sz_pix = 21
    else:
        par_sz_pix = args.particle_size
    par_sz = par_sz_pix / sz
    heads_dict = {"10077": 0, "10078": 1, "10081": 2, "pdb_6bqv": 3, "ss_1": 4,
                  "pdb_6bhu": 5, "pdb_6bcx": 6, "pdb_6bcq": 7, "pdb_6bco": 8,
                  "pdb_6az1": 9, "pdb_5y6p": 10, "pdb_5xwy": 11, "pdb_5w3s": 12,
                  "pdb_5ngm": 13, "pdb_5mmi": 14, "pdb_5foj": 15, "pdb_4zor": 16,
                  "pdb_3j9i": 17, "pdb_2gtl": 18, "pdb_1sa0": 19, "lf_1": 20,
                  "hh_2": 21, "gk_1": 22, "10156": 23, "10153": 24, "10122": 25,
                  "10097": 26, "10089": 27, "10084": 28, "10017": 29,
                  "pdb_6b7n": 30, "pdb_6b44": 31, "pdb_5xnl": 32, "pdb_5w3l": 33,
                  "pdb_5vy5": 34, "pdb_4hhb": 35, "pdb_2wri": 36, "gen": 37, "fine_tuned": 38}
    prediction_head = args.prediction_head
    prediction_subfolder = args.prediction_subfolder
    prediction_conf = args.prediction_conf
    evaluate_format = args.evaluate_format
    cryolo_model = args.cryolo_model
    boxnet_suffix = args.boxnet_suffix
    image_name = args.image_name
    total_reference = 0
    total_reference_neg = 0
    detections = []
    prec = None
    rec = None
    ap = None
    fpr = None
    auroc = None
    rec3 = None

    ############ Useful paths
    PATH = Path("data")
    PATH2 = Path("data")
    path3 = "data/" + source_list + "/"
    IMAGES = source_image

    with open(path3+'uri_list.pickle', 'rb') as handle:
        uri_list = pickle.load(handle)

    with open(path3+'centroids_list.pickle', 'rb') as handle:
        centroids_list = pickle.load(handle)

    splits_dict = dict()
    lens_dict_tst = dict()
    lens_dict_vld = dict()
    uri_list_tst_dict = dict()
    centroids_list_tst_dict = dict()
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
        
            uri_list_tst_dict[name] = [uri_list[i] for i in splits_dict[name][:lens_dict_tst[name]]]
            centroids_list_tst_dict[name] = [centroids_list[i] for i in splits_dict[name][:lens_dict_tst[name]]]
    except:
        print("Error reading splits file!")
        exit(1)

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

    ######### Making anchor boxes
    k = 1

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

    aux.aux.Tparticle[0] = heads_dict[prediction_head] * torch.eye(1, dtype=torch.int8)

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

    def reset_metrics(pthresh=0, dthresh=0):
        nonlocal total_reference
        nonlocal total_reference_neg
        nonlocal detections
        nonlocal prec
        nonlocal rec
        nonlocal ap
        nonlocal fpr
        nonlocal auroc
        nonlocal rec3

        total_reference = 0
        total_reference_neg = 0
        detections = []
        prec = None
        rec = None
        ap = None
        fpr = None
        auroc = None
        rec3 = None

    ######### Making anchor boxes

    def get_y_tst(cent, sizes):
        """ gets rid of any of the bounding boxes that are just padding """
        cent = torch.cat([(cent.float().view(-1, 2)[:, 0] / sizes[0]).unsqueeze(1),
                   (cent.float().view(-1, 2)[:, 1] / sizes[1]).unsqueeze(1)], 1)
        resulting_cents = cent[(cent[:, 0] > 0) | (cent[:, 1] > 0)]
        return resulting_cents, V(torch.ones(len(resulting_cents)))

    def calc_metrics2(pred, targ, md=None, x0=None, clas_pr=None, five_crop=False, sizes=None, par_size=None, tpcenters=False, IOU_thresh=iou_tst, conf_thresh=prediction_conf, image_name=None):
        nonlocal total_reference
        nonlocal total_reference_neg
        nonlocal par_sz
        nonlocal detections

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

            # prediction calcs
            cent, clas = get_y_tst(cent, sizes)
            clas_pr = b_clas
            clas_ids = np.ones(len(b_clas), dtype=float)

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
                    if tpcenters:
                        TPcenters[k] = 1
            total_reference += len(cent)
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
                        if tpc > 0:
                            draw_rect(ax, b, color=colr_list[int(c) + 1], lw=4)
                    for i, (b, c, pr, tpc) in enumerate(zip(bb2, clas2, prs2, TPcenters2)):
                        if tpc > 0:
                            draw_rect(ax, b, color='red', lw=2, outline=2)

                def torch_gt(ax, ima, bbox1, clas1, bbox2, clas2, sizes, prs=None, thresh=0.4, TPcenters=None):
                    bbox1 = np.transpose(
                        np.array(to_np([(bbox1[:, 0] * sizes[0]).long(), (bbox1[:, 1] * sizes[1]).long()])))
                    bbox2 = np.transpose(
                        np.array(to_np([(bbox2[:, 0] * sizes[0]).long(), (bbox2[:, 1] * sizes[1]).long()])))
                    return show_ground_truth(ax, ima, bbox1, bbox2, sizes, to_np(clas1), to_np(clas2),
                                             to_np(prs) if prs is not None else None, thresh, TPcenters)

                def plot_results(thresh, x0, sizes, md, image_name, b_cent1=b_cent, clas_ids1=clas_ids, b_cent2=b_cent, clas_ids2=clas_ids, TPcenters=None):
                    x0 = to_np(x0)
                    fig, axes = plt.subplots(1, 1, figsize=(sizes[0]/25.0, sizes[1]/25.0))#figsize=(50.8, 49.2))
                    ima = md.val_ds.ds.denorm(x0)[0]
                    torch_gt(axes, ima.squeeze(), b_cent1, clas_ids1, b_cent2, clas_ids2, sizes, None,
                             clas_pr.max() * thresh, TPcenters)
                    plt.tight_layout()
                    fig.savefig(image_name + '_' + str(np.random.randint(10, 10000)) + ".pdf")

                import matplotlib.cm as cmx
                import matplotlib.colors as mcolors
                from cycler import cycler

                def get_cmap(N):
                    color_norm = mcolors.Normalize(vmin=0, vmax=N - 1)
                    return cmx.ScalarMappable(norm=color_norm, cmap='Set3').to_rgba

                num_colr = 5
                cmap = get_cmap(num_colr)
                colr_list = [cmap(float(x)) for x in range(num_colr)]

                plot_results(.5, x0, sizes, md, image_name, b_cent1=cent,
                             clas_ids1=np.asarray(clas.cpu().numpy(), dtype=np.int), b_cent2=b_cent, TPcenters=TPcenters)
        return 1

    def getKey(item):
        return item[0]

    def CalculateAveragePrecision(prec, rec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    def CalculateAUROC(rec, fpr):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(rec[-1])
        mfpr = []
        mfpr.append(0)
        [mfpr.append(e) for e in fpr]
        mfpr.append(1)
        ii = []
        for i in range(len(mfpr) - 1):
            if mfpr[1:][i] != mfpr[0:-1][i]:
                ii.append(i + 1)
        auroc = 0
        for i in ii:
            auroc = auroc + np.sum((mfpr[i] - mfpr[i - 1]) * (mrec[i]))
        return [auroc, mrec[0:len(mfpr)], mfpr[0:len(mfpr)], ii]

    def finalize_metrics(interpolation=True):
        nonlocal total_reference
        nonlocal total_reference_neg
        nonlocal detections
        nonlocal prec
        nonlocal rec
        nonlocal rec3
        nonlocal fpr
        nonlocal ap
        nonlocal auroc

        detections = sorted(detections, key=getKey)[::-1]
        acc_TP = 0
        acc_FP = 0
        for i in range(len(detections)):
            acc_TP += detections[i][1]
            acc_FP += detections[i][2]
            detections[i][3] = np.divide(acc_TP, (acc_TP + acc_FP)) # precision
            detections[i][4] = np.divide(acc_TP, total_reference)  # recall
            detections[i][5] = np.divide(acc_FP, total_reference_neg)  # FPR

        if interpolation:
            [ap, prec, rec, ii] = CalculateAveragePrecision(np.asarray(detections)[:,3], np.asarray(detections)[:,4])
            [auroc, rec3, fpr, ii2] = CalculateAUROC(np.asarray(detections)[:, 4], np.asarray(detections)[:, 5])
        else:
            print("Error: Feature not implemented!")
            exit(1)

    def precision():
        nonlocal prec
        return prec

    def recall():
        nonlocal rec
        return rec

    def receiver_operating_characteristic():
        nonlocal fpr
        nonlocal auroc
        nonlocal rec3
        return auroc, fpr, rec3

    def calc_auc(rec2, prec2, avgs2=None):
        auc = 0
        avgdis_auc = 0

        prec_at_rec90 = np.zeros(2)
        prec_aft_rec90 = np.zeros(2)
        prec_aft_rec90[0] = 1
        prec_bef_rec90 = np.zeros(2)
        if rec2[0] == 0.9:
            prec_at_rec90[0] = rec2[0]
            prec_at_rec90[1] = prec2[0]

        rec_at_prec90 = 0
        if prec2[0] == 0.9:
            rec_at_prec90 = rec2[0]

        for i in range(len(rec2) - 1):
            auc += ((prec2[i] + prec2[i + 1]) / 2) * (rec2[i + 1] - rec2[i])

            if rec2[i + 1] == 0.9:
                if prec2[i + 1] > prec_at_rec90[1]:
                    prec_at_rec90[0] = rec2[i + 1]
                    prec_at_rec90[1] = prec2[i + 1]
            elif rec2[i + 1] > 0.9:
                if rec2[i + 1] < prec_aft_rec90[0]:
                    prec_aft_rec90[0] = rec2[i + 1]
                    prec_aft_rec90[1] = prec2[i + 1]
            elif rec2[i + 1] < 0.9:
                if rec2[i + 1] > prec_bef_rec90[0]:
                    prec_bef_rec90[0] = rec2[i + 1]
                    prec_bef_rec90[1] = prec2[i + 1]

            if prec2[i + 1] == 0.9:
                if rec2[i + 1] > rec_at_prec90:
                    rec_at_prec90 = rec2[i + 1]
            elif (prec2[i + 1] > 0.9) and (prec2[i] < 0.9):
                range_bef = 0.9 - prec2[i]
                range_aft = prec2[i + 1] - 0.9
                rec_at_prec90_temp = (range_bef * rec2[i + 1] + range_aft * rec2[i]) / (range_bef + range_aft)
                if rec_at_prec90_temp > rec_at_prec90:
                    rec_at_prec90 = rec_at_prec90_temp
            elif (prec2[i + 1] < 0.9) and (prec2[i] > 0.9):
                range_bef = prec2[i] - 0.9
                range_aft = 0.9 - prec2[i + 1]
                rec_at_prec90_temp = (range_bef * rec2[i + 1] + range_aft * rec2[i]) / (range_bef + range_aft)
                if rec_at_prec90_temp > rec_at_prec90:
                    rec_at_prec90 = rec_at_prec90_temp

        prec_at_rec90_temp = 0
        if (prec_aft_rec90[0] < 1) and (prec_bef_rec90[0] > 0):
            range_bef = np.abs(0.9 - prec_bef_rec90[0])
            range_aft = np.abs(prec_aft_rec90[0] - 0.9)
            prec_at_rec90_temp = (range_bef * prec_aft_rec90[1] + range_aft * prec_bef_rec90[1]) / (
                    range_bef + range_aft)
        if prec_at_rec90[1] < prec_at_rec90_temp:
            prec_at_rec90[1] = prec_at_rec90_temp

        return auc, prec_at_rec90[1], rec_at_prec90
    
    def add_border(x, y):
        sizes = torch.tensor(x.shape[2:])
        rmdr = torch.fmod(16 - torch.fmod(sizes, 16), 16)
        if torch.sum(rmdr) > 0:

            left = rmdr[0] / 2
            up = rmdr[1] / 2

            sizes_padded = sizes + rmdr
            x_padded = torch.zeros((1, 1, sizes_padded[0], sizes_padded[1]))
            x_padded[0, 0, left:left + sizes[0], up:up + sizes[1]] = x[0][0]

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

    target_uri_list_tst = []
    target_centroids_list_tst = []
    target_tst_idxs = ()
    target_tst_idxs_index = 0
    for c1 in target_datasets:
        target_uri_list_tst.extend(uri_list_tst_dict[c1])
        target_centroids_list_tst.extend(centroids_list_tst_dict[c1])
        target_tst_idxs = target_tst_idxs + tuple(
            range(target_tst_idxs_index, target_tst_idxs_index + lens_dict_tst[c1]))
        target_tst_idxs_index += len(uri_list_tst_dict[c1])

    # # Gen head on targets:
    fnames_dict = [target_uri_list_tst[i][:-4] for i in range(len(target_uri_list_tst))]
    centroids_dict = [target_centroids_list_tst[i][1:] for i in range(len(target_uri_list_tst))]
    df = pd.DataFrame({'fnames': fnames_dict, 'centroids': centroids_dict}, columns=['fnames', 'centroids'])
    df.to_csv(path3 + "centroids_" + str(len(target_datasets)) + ".csv", index=False)
    val_idxs = target_tst_idxs
    CENT_CSV_TARGET_DATASETS = Path(PATH2, source_list + "/centroids_" + str(len(target_datasets)) + ".csv")

    aux.aux.Tparticle[0] = heads_dict[prediction_head] * torch.eye(1, dtype=torch.int8)

    md_target_datasets_shared_tst0 = ImageClassifierData.from_csv(path=PATH, folder=IMAGES,
                                                                    csv_fname=CENT_CSV_TARGET_DATASETS,
                                                                    val_idxs=val_idxs, tfms=tfms0, bs=1,
                                                                    suffix='.tif', continuous=True,
                                                                    num_workers=16)
    val_ds2 = ConcatLblDataset2(md_target_datasets_shared_tst0.val_ds, heads_dict["gen"])
    md_target_datasets_shared_tst0.val_dl.dataset = val_ds2

    learn = ConvLearner(md_target_datasets_shared_tst0, models)
    learn.model.cuda()
    
    if load is not None:
        learn.load(load)

    if prediction_head not in heads_dict:
            print("Error: Prediction head undefined!")
            exit(1)

    iter0 = iter(md_target_datasets_shared_tst0.val_dl)

    learn.model.eval()
    aux.aux.heads_eval_mode(learn.model[1])
    start_time = time.time()
    reset_metrics()

    with torch.no_grad():
        for val_counter in range(len(val_idxs)):

            if evaluate_format == "star_cryolo":
                learn.set_data(md_target_datasets_shared_tst0)
                x0, y0 = next(iter0)
                x0 = V(x0)
                y0[0] = y0[0].float()
                y0 = V(y0)

                with open("data/predictions/out_" + cryolo_model + '/STAR/' + fnames_dict[val_counter] + ".star", "r") as text_file:
                # with open("data/predictions/" + fnames_dict[val_counter] + "_BoxNet2Mask_20180918.star", "r") as text_file:
                    for i in range(10):
                        input_line = text_file.readline()
                    predicted_centroids = []
                    predicted_confs = []
                    while input_line != '':
                        line_content = []
                        for i in input_line.split('\t'):
                            if i != '':
                                line_content.append(i)
                        predicted_centroids.append([float(line_content[0]), float(line_content[1])])
                        predicted_confs.append(float(line_content[4][:-2]))
                        input_line = text_file.readline()
                    predicted_centroids = np.array(predicted_centroids, dtype=np.float32).reshape(1, -1, 2)
                    predicted_centroids[0, :, 0] = (predicted_centroids[0, :, 0]) / (x0.shape[3])


                    if fnames_dict[val_counter][:-3] == "empiar_10081" or fnames_dict[val_counter][:-2] == "empiar_10081":
                        predicted_centroids[0, :, 1] = (x0.shape[2] - predicted_centroids[0, :, 1] - 0) / (x0.shape[2])
                    elif fnames_dict[val_counter][:-3] == "empiar_10097" or fnames_dict[val_counter][:-2] == "empiar_10097":
                        predicted_centroids[0, :, 1] = (x0.shape[2] - predicted_centroids[0, :, 1] - 0) / (x0.shape[2])
                    elif fnames_dict[val_counter][:-3] == "empiar_10153" or fnames_dict[val_counter][:-2] == "empiar_10153":
                        predicted_centroids[0, :, 1] = (x0.shape[2] - predicted_centroids[0, :, 1] - 0) / (x0.shape[2])
                    elif fnames_dict[val_counter][:-3] == "empiar_10156" or fnames_dict[val_counter][:-2] == "empiar_10156":
                        predicted_centroids[0, :, 1] = (x0.shape[2] - predicted_centroids[0, :, 1] - 0) / (x0.shape[2])
                    else:
                        predicted_centroids[0, :, 1] = (x0.shape[3] - predicted_centroids[0, :, 1]) / (x0.shape[2])

                    predicted_centroids[0, :, [0, 1]] = predicted_centroids[0, :, [1, 0]]
                    predicted_centroids = T(predicted_centroids)
                    predicted_confs = np.array(predicted_confs, dtype=np.float32).reshape(1, -1)

                    pred0 = [predicted_confs, predicted_centroids]

                if not draw:
                    calc_metrics2(pred0, y0, five_crop=False, sizes=x0.shape[2:])
                else:
                    calc_metrics2(pred0, y0, x0=x0, md=md_target_datasets_shared_tst0, five_crop=False,
                                sizes=x0.shape[2:], tpcenters=True, image_name=image_name)
            elif evaluate_format == "star_boxnet":
                learn.set_data(md_target_datasets_shared_tst0)
                x0, y0 = next(iter0)
                x0 = V(x0)
                y0[0] = y0[0].float()
                y0 = V(y0)

                # with open("data/predictions/" + fnames_dict[val_counter] + "_BoxNet2_20180602.star", "r") as text_file:
                # with open("data/predictions/" + fnames_dict[val_counter] + "_BoxNet2Mask_20180918.star", "r") as text_file:
                # with open("data/predictions/" + fnames_dict[val_counter] + "_BoxNet_blank_trn4.star", "r") as text_file:
                with open("data/predictions/" + fnames_dict[val_counter] + boxnet_suffix, "r") as text_file:
                    for i in range(12):
                        input_line = text_file.readline()
                    predicted_centroids = []
                    predicted_confs = []
                    while input_line != '':
                        line_content = []
                        for i in input_line.split(' '):
                            if i != '':
                                line_content.append(i)
                        predicted_centroids.append([float(line_content[0]), float(line_content[1])])
                        predicted_confs.append(float(line_content[3]))
                        input_line = text_file.readline()
                    predicted_centroids = np.array(predicted_centroids, dtype=np.float32).reshape(1, -1, 2)
                    predicted_centroids[0, :, 0] = predicted_centroids[0, :, 0] / x0.shape[3]
                    predicted_centroids[0, :, 1] = predicted_centroids[0, :, 1] / x0.shape[2]
                    predicted_centroids[0, :, [0, 1]] = predicted_centroids[0, :, [1, 0]]
                    predicted_centroids = T(predicted_centroids)
                    predicted_confs = np.array(predicted_confs, dtype=np.float32).reshape(1, -1)

                    pred0 = [predicted_confs, predicted_centroids]
                if not draw:
                    calc_metrics2(pred0, y0, five_crop=False, sizes=x0.shape[2:])
                else:
                    calc_metrics2(pred0, y0, x0=x0, md=md_target_datasets_shared_tst0, five_crop=False,
                                sizes=x0.shape[2:], tpcenters=True, image_name=image_name)
            elif evaluate_format == "star":
                learn.set_data(md_target_datasets_shared_tst0)
                x0, y0 = next(iter0)
                x0, y0 = add_border(x0, y0)
                x0 = V(x0)
                y0[0] = y0[0].float()
                y0 = V(y0)

                with open("data/predictions/"+prediction_subfolder+'/'+fnames_dict[val_counter]+".star", "r") as text_file:
                    for i in range(7):
                        input_line = text_file.readline()
                    predicted_centroids = []
                    predicted_confs = []
                    while input_line != '' and input_line != '\n':
                        line_content = []
                        for i in input_line.split('\n')[0].split('\t'):
                            if i != '':
                                line_content.append(i)
                        predicted_centroids.append([float(line_content[0]), float(line_content[1])])
                        predicted_confs.append(float(line_content[2]))
                        input_line = text_file.readline()
                    predicted_centroids = np.array(predicted_centroids, dtype=np.float32).reshape(1, -1, 2)
                    predicted_centroids[0, :, 0] = predicted_centroids[0, :, 0] / x0.shape[2]
                    predicted_centroids[0, :, 1] = predicted_centroids[0, :, 1] / x0.shape[3]
                    predicted_centroids = T(predicted_centroids)
                    predicted_confs = np.array(predicted_confs, dtype=np.float32).reshape(1, -1)

                    pred0 = [predicted_confs, predicted_centroids]
                if not draw:
                    calc_metrics2(pred0, y0, five_crop=False, sizes=x0.shape[2:])
                else:
                    calc_metrics2(pred0, y0, x0=x0, md=md_target_datasets_shared_tst0, five_crop=False,
                                sizes=x0.shape[2:], tpcenters=True, image_name=image_name)
            else:
                print("Error: Evaluation input file format not implemented!")
                exit(1)

    finalize_metrics()

    print('val_time: ', time.time() - start_time)

    rec2 = recall()
    prec2 = precision()
    auroc2, fpr2, rec3 = receiver_operating_characteristic()                
    auc, prec_at_rec90, rec_at_prec90 = calc_auc(rec2, prec2)

    print("Gen head on targets:")
    print("AP:                          ", auc)
    print("Precision at recall=90:      ", prec_at_rec90)
    print("Recall at precision=90:      ", rec_at_prec90)
    print("AUROC:                       ", auroc2)
    print("\n")

def main(argv=None):
    start()
    exit(0)


if __name__ == '__main__':
    main()