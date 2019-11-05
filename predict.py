from pathlib import Path
import time
from copy import deepcopy as dc
import argparse

from fastai.conv_learner import *
from fastai.dataset import *

from fastai.resnet_29_6 import resnet3 as myresnet

def start():
    # check to make sure you set the device
    # cuda_id = 0
    # torch.cuda.set_device(cuda_id)
    # new edit test7
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
                    predicted_star = 'data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnAutopickFigureOfMerit #3\n'
                    for k in range(len(clas_pr[::-1])):
                        if clas_ids[k] == 1 and clas_pr[k] > prediction_conf:
                            predicted_centroids = ','.join(
                                [predicted_centroids, str(b_cent[k][0]), str(b_cent[k][1])])
                            predicted_confs = ','.join([predicted_confs, str(clas_pr[k])])
                            predicted_star = predicted_star +  str(b_cent[k][0]) + '\t' + str(b_cent[k][1]) + '\t' + str(clas_pr[k]) + '\n'
                    predicted_centroids = predicted_centroids[1:]
                    predicted_confs = predicted_confs[1:]

                    if prediction_subfolder:
                        with open("data/boxnet/predictions/"+prediction_subfolder+'/'+fnames_dict[val_counter]+"_coords.txt", "w") as text_file:
                            print(predicted_centroids, file=text_file)
                        with open("data/boxnet/predictions/"+prediction_subfolder+'/'+fnames_dict[val_counter]+"_confs.txt", "w") as text_file:
                            print(predicted_confs, file=text_file)
                        with open("data/boxnet/predictions/"+prediction_subfolder+'/'+fnames_dict[val_counter]+".star", "w") as text_file:
                            print(predicted_star, file=text_file)
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
        exit(1)

def main(argv=None):
    start()
    exit(0)


if __name__ == '__main__':
    main()