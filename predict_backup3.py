from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path

import time
from copy import deepcopy as dc

import argparse

from fastai.resnet_29_6 import resnet3 as myresnet

def start():
    # check to make sure you set the device
    # cuda_id = 0
    # torch.cuda.set_device(cuda_id)
    version = '29_6'

    parser = argparse.ArgumentParser(description='A cross dataset generalization study using 37 Cryo-EM datasets.')
    parser.add_argument('-t', '--training_type', required=True, type=str,
                        choices=['2b', '3b', '4b', '5b', '3c', '4c', '5c'], dest='training_type')
    parser.add_argument('-l', '--load', type=str, required=True, default=None, dest='load')
    parser.add_argument('-lv', '--load_version', type=str, default=version, dest='load_version')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, dest='bs')
    parser.add_argument('-td', '--target_datasets',
                        default='pdb_6b7n$pdb_6b44$pdb_5xnl$pdb_5w3l$pdb_5vy5$pdb_4hhb$pdb_2wri', type=str,
                        dest='target_dataset')
    parser.add_argument('-cs', '--crop_size', type=int, default=368, dest='crop_size')  # up to 240 for original images
    parser.add_argument('-si', '--source_image', type=str, default='trainingdata11_2', dest='source_image')
    parser.add_argument('-sl', '--source_list', type=str, default='labels12_2', dest='source_list')
    parser.add_argument('-d', '--drive', default='ssd', type=str, choices=['ssd', 'hdd'], dest='drive')
    parser.add_argument('-ph', '--prediction_head', type=str, default='gen', dest='prediction_head')
    parser.add_argument('-psf', '--prediction_subfolder', type=str, default=None, dest='prediction_subfolder')
    parser.add_argument('-pc', '--prediction_conf', type=float, default=0, dest='prediction_conf')
    args = parser.parse_args()


    ############ Mainly used variables
    training_type = args.training_type
    load = args.load
    load_version = args.load_version
    source_image = args.source_image
    source_list = args.source_list
    bs = args.bs
    target_datasets = args.target_dataset.split('$')

    f_model = myresnet
    sz = args.crop_size

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

    prediction_head = args.prediction_head

    prediction_subfolder = args.prediction_subfolder

    prediction_conf = args.prediction_conf

    ############ Useful paths
    drive = args.drive
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

    lens_dict = {"10077": 3, "10078": 3, "10081": 3, "pdb_6bqv": 2, "ss_1": 3,
                 "pdb_6bhu": 2, "pdb_6bcx": 2, "pdb_6bcq": 2, "pdb_6bco": 2,
                 "pdb_6az1": 2, "pdb_5y6p": 3, "pdb_5xwy": 2, "pdb_5w3s": 2,
                 "pdb_5ngm": 2, "pdb_5mmi": 2, "pdb_5foj": 1, "pdb_4zor": 2,
                 "pdb_3j9i": 2, "pdb_2gtl": 2, "pdb_1sa0": 2, "lf_1": 3,
                 "hh_2": 3, "gk_1": 3, "10156": 3, "10153": 3, "10122": 3,
                 "10097": 3, "10089": 3, "10084": 3, "10017": 3, "pdb_6b7n": 2,
                 "pdb_6b44": 2, "pdb_5xnl": 2, "pdb_5w3l": 2, "pdb_5vy5": 1,
                 "pdb_4hhb": 1, "pdb_2wri": 2}


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

    auxilary.auxilary.Tparticle[0] = heads_dict[prediction_head] * torch.eye(1, dtype=torch.int8)

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

        learn = ConvLearner(md_target_datasets_shared_tst0, models)
        learn.model.cuda()

        if load is not None:
            learn.load('SSPicker_' + load_version + '_' + training_type + '_' + load)

        if prediction_head not in heads_dict:
            print("Error: Prediction head undefined!")
            exit(1)

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
                    predicted_star = 'data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n'
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
                        with open("data/boxnet/predictions/"+prediction_subfolder+'/'+fnames_dict[val_counter]+".star", "w") as text_file:
                            print('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n', file=text_file)
                            print('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n', file=text_file)
                    else:
                        # pd.DataFrame(np_array).to_csv("path/to/file.csv")
                        with open("data/boxnet/predictions/"+fnames_dict[val_counter]+"_coords.txt", "w") as text_file:
                            print(predicted_centroids, file=text_file)
                        with open("data/boxnet/predictions/"+fnames_dict[val_counter]+"_confs.txt", "w") as text_file:
                            print(predicted_confs, file=text_file)

                    # with open("data/boxnet/predictions/"+fnames_dict[val_counter]+".txt", 'r') as text_file:
                    #     x = text_file.read().split(',')

        print('prediction_time: ', time.time() - start_time)

    else:
        print("Error: Prediction model type not supported!")

def main(argv=None):
    start()
    exit(0)


if __name__ == '__main__':
    main()