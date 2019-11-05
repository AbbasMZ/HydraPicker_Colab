import argparse
import datetime
import difflib
import json
import multiprocessing
import os

import numpy as np
import terminaltables
from keras.utils import multi_gpu_model

from cryolo import predict
from courses.dl2 import config_tools
from courses.dl2 import imagereader
from .frontend import YOLO

"""
This scripts evaluates a model based on ground truth box files.

Per annotation folder it calculates:
    As table:
    - Optimal threshold based on F1 score
    - AUC 
    - Precision using optimal threshold, 0.3, 0.2 and 0.1
    - Recall using optimal threshold, 0.3, 0.2 and 0.1
    - F1 score using optimal threshold, 0.3, 0.2 and 0.1

Morover it calculates for the full data:

    - Optimal threshold based on F1 score
    - AUC 
    As table 2:
    - Precision using optimal threshold, 0.3, 0.2 and 0.1
    - Recall using optimal threshold, 0.3, 0.2 and 0.1
    - F1 score using optimal threshold, 0.3, 0.2 and 0.1




"""

argparser = argparse.ArgumentParser(
    description="Calculate common statistics using box files"
)

argparser.add_argument("-c", "--config", required=True, help="path to config")

argparser.add_argument("-w", "--weights", required=True, help="path to trained model")

argparser.add_argument(
    "-i", "--images", help="path folder with test images"
)

argparser.add_argument(
    "-b", "--boxfiles", help="path to folder with box files (ground truth)"
)

argparser.add_argument("-r", "--runfile", help="Evaluate based on runfile")

argparser.add_argument(
    "-g", "--gpu", default=0, type=int, help="Specifiy which gpu should be used."
)


def _main_():
    args = argparser.parse_args()
    path_config = args.config
    path_weights = args.weights
    path_runfile = None
    if args.runfile:
        path_runfile = args.runfile
        path_boxfiles = None
        path_images = None
    else:
        if args.boxfiles is not None and args.images is not None:
            path_boxfiles = args.boxfiles
            path_images = args.images
        else:
            import sys

            sys.exit(
                "Please specify either the path to your validation data with -b (--boxfiles) and -i (--images)"
                " or the path to the runfile -r (--runfile)"
            )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Create evaluation datasets
    datasets = prepare_datasets(
        path_config, path_weights, path_boxfiles, path_runfile, path_images
    )

    # Calculate statistics
    calculate_all_statistics(datasets)

    # Calculate optimal F1 F2 threshold
    arange = np.arange(start=0, stop=1.001, step=0.001)
    f1_array = np.zeros(len(arange))
    f1_N_array = np.zeros(len(arange))
    f2_array = np.zeros(len(arange))
    f2_N_array = np.zeros(len(arange))
    for foldername, dataset in datasets.items():
        if "results" in dataset:
            result_dict = dataset["results"]
            for i, thr in enumerate(arange):
                if str(thr) in result_dict:
                    f1_array[i] += result_dict[str(thr)][0]
                    f1_N_array[i] += 1
                    f2_array[i] += result_dict[str(thr)][5]
                    f2_N_array[i] += 1
    for i in range(len(f1_array)):
        if f1_N_array[i] > 0:
            f1_array[i] = f1_array[i] / f1_N_array[i]
        if f2_N_array[i] > 0:
            f2_array[i] = f2_array[i] / f2_N_array[i]
    best_f1_thresh_index = np.argmax(f1_array)
    best_f1_thresh = result_dict["THRESH_LIST"][best_f1_thresh_index]

    best_f2_thresh_index = np.argmax(f2_array)
    best_f2_thresh = result_dict["THRESH_LIST"][best_f2_thresh_index]

    # Print statistics
    table_data = []
    template_column_folder = "\033[1;31m{0}\033[0m"
    template_column_value = "\033[1;34m{0}\033[0m"
    template_row_value = "\033[100m{0}\033[0m"
    template_row_white = "\033[0m{0}\033[0m"
    template = template_row_value
    headings = [
        "AUC",
        "Topt",
        "R\nTopt",
        "R\n0.3",
        "R\n0.2",
        # "R\n0.1",
        "P\nTopt",
        "P\n0.3",
        "P\n0.2",
        # "P\n0.1",
        "F1\nTopt",
        "F1\n0.3",
        "F1\n0.2",
        # "F1\n0.1"
        "IOU\nTopt",
        "IOU\n0.3",
        "IOU\n0.2",
    ]
    headings = [
        "\n".join(
            [template_column_value.format(entry2) for entry2 in entry.split("\n")]
        )
        if index % 2 == 1
        else entry
        for index, entry in enumerate(headings)
    ]
    headings.insert(0, template_column_folder.format("Folder"))
    table_data.append(headings)
    n = 0
    auc_avg = 0
    topt_avg = 0
    pTopt_avg = 0
    p03_avg = 0
    p02_avg = 0
    p01_avg = 0
    rTopt_avg = 0
    r03_avg = 0
    r02_avg = 0
    r01_avg = 0
    fTopt_avg = 0
    f03_avg = 0
    f02_avg = 0
    f01_avg = 0
    iouTopt_avg = 0
    iou03_avg = 0
    iou02_avg = 0

    for foldername, dataset in datasets.items():
        result_dict = dataset["results"]

        AUC = "{0:.2f}".format(result_dict["AUC"])
        auc_avg += result_dict["AUC"]

        topt = "{0:.2f}".format(result_dict["MAX_F1_THRESH"])
        topt_avg += result_dict["MAX_F1_THRESH"]

        pTopt = "{0:.2f}".format(result_dict[str(result_dict["MAX_F1_THRESH"])][1])
        pTopt_avg += result_dict[str(result_dict["MAX_F1_THRESH"])][1]

        p03 = "{0:.2f}".format(result_dict[str(0.3)][1])
        p03_avg += result_dict[str(0.3)][1]

        p02 = "{0:.2f}".format(result_dict[str(0.2)][1])
        p02_avg += result_dict[str(0.2)][1]

        p01 = "{0:.2f}".format(result_dict[str(0.1)][1])
        p01_avg += result_dict[str(0.1)][1]

        rTopt = "{0:.2f}".format(result_dict[str(result_dict["MAX_F1_THRESH"])][2])
        rTopt_avg += result_dict[str(result_dict["MAX_F1_THRESH"])][2]

        r03 = "{0:.2f}".format(result_dict[str(0.3)][2])
        r03_avg += result_dict[str(0.3)][2]

        r02 = "{0:.2f}".format(result_dict[str(0.2)][2])
        r02_avg += result_dict[str(0.2)][2]

        r01 = "{0:.2f}".format(result_dict[str(0.1)][2])
        r01_avg += result_dict[str(0.1)][2]

        fTopt = "{0:.2f}".format(result_dict[str(result_dict["MAX_F1_THRESH"])][0])
        fTopt_avg += result_dict[str(result_dict["MAX_F1_THRESH"])][0]

        f03 = "{0:.2f}".format(result_dict[str(0.3)][0])
        f03_avg += result_dict[str(0.3)][0]

        f02 = "{0:.2f}".format(result_dict[str(0.2)][0])
        f02_avg += result_dict[str(0.2)][0]

        f01 = "{0:.2f}".format(result_dict[str(0.1)][0])
        f01_avg += result_dict[str(0.1)][0]

        iouTopt = "{0:.2f}".format(result_dict[str(result_dict["MAX_F1_THRESH"])][3])
        iouTopt_avg += result_dict[str(result_dict["MAX_F1_THRESH"])][3]

        iou03 = "{0:.2f}".format(result_dict[str(0.3)][3])
        iou03_avg += result_dict[str(0.3)][3]

        iou02 = "{0:.2f}".format(result_dict[str(0.2)][3])
        iou02_avg += result_dict[str(0.2)][3]

        table_row = [
            AUC,
            topt,
            rTopt,
            r03,
            r02,
            # r01,
            pTopt,
            p03,
            p02,
            # p01,
            fTopt,
            f03,
            f02,
            # f01
            iouTopt,
            iou03,
            iou02,
        ]

        table_row = [
            "\n".join(
                [
                    template.format(template_column_value.format(entry2))
                    for entry2 in entry.split("\n")
                ]
            )
            if index % 2 == 1
            else template.format(entry)
            for index, entry in enumerate(table_row)
        ]
        table_row.insert(0, template.format(template_column_folder.format(foldername)))

        table_data.append(table_row)

        if template == template_row_value:
            template = template_row_white
        else:
            template = template_row_value
        n = n + 1
    table_row = [
        "{0:.2f}".format(auc_avg / n),
        "{0:.2f}".format(topt_avg / n),
        "{0:.2f}".format(rTopt_avg / n),
        "{0:.2f}".format(r03_avg / n),
        "{0:.2f}".format(r02_avg / n),
        # '{0:.2f}'.format(r01_avg/n),
        "{0:.2f}".format(pTopt_avg / n),
        "{0:.2f}".format(p03_avg / n),
        "{0:.2f}".format(p02_avg / n),
        # '{0:.2f}'.format(p01_avg/n),
        "{0:.2f}".format(fTopt_avg / n),
        "{0:.2f}".format(f03_avg / n),
        "{0:.2f}".format(f02_avg / n),
        # '{0:.2f}'.format(f01_avg/n)
        "{0:.2f}".format(iouTopt_avg / n),
        "{0:.2f}".format(iou03_avg / n),
        "{0:.2f}".format(iou02_avg / n),
    ]
    table_row = [
        "\n".join(
            [
                template.format(template_column_value.format(entry2))
                for entry2 in entry.split("\n")
            ]
        )
        if index % 2 == 1
        else template.format(entry)
        for index, entry in enumerate(table_row)
    ]
    table_row.insert(0, template.format(template_column_folder.format("AVERAGE")))

    table_data.append(table_row)

    table = terminaltables.AsciiTable(table_data)
    import hashlib

    hasher = hashlib.md5()
    with open(path_weights, "rb") as afile:
        buf = afile.read()
        hasher.update(buf)
    md5sum = hasher.hexdigest()

    print("##########################")
    print("### Summary evaluation ###")
    print("##########################")
    print("Date: ", datetime.date.today())
    print("Model: ", path_weights, "Model checksum (MD5):", str(md5sum))
    if args.runfile:
        print("Config:", path_config, "Runefile:", path_runfile)
    else:
        print(
            "Config:",
            path_config,
            "Valid images:",
            path_images,
            "Valid annotations: ",
            path_boxfiles,
        )
    print("R = Recall. Higher is better. Range 0 - 1")
    print("P = Precision. Higher is better. Range 0 -1")
    print("F1 = F1 Score, harmonic mean of R and P. Higher is better. Range 0 - 1")
    print("F2 = Same as F1 Score, but put higher weights on R (recall). Range 0 - 1")
    print("AUC = Area under precision recall curve. Higher is better. Range 0 - 1")
    print("Topt = Optimal confidence threshold at maximum F1 score")
    print(
        "IOU = Intersection over Union, as higher the better the centering. Range 0 - 1"
    )
    print(table.table)
    print("##########################")
    print("Best confidence threshold ( -t ) according F1 statistic: ", best_f1_thresh)
    print("Best confidence threshold ( -t ) according F2 statistic: ", best_f2_thresh)
    print("##########################")


def get_anchors(config, path_weights):
    is_single_anchor = config["model"]["anchors"] == 2
    grid_w, grid_h = config_tools.get_gridcell_dimensions(config)
    anchors = None
    import h5py
    with h5py.File(path_weights, mode='r') as f:
        try:
            anchors = list(f["anchors"])
        except KeyError:
            anchors = None

    if is_single_anchor and anchors is None:
        num_patches = 1
        if "num_patches" in config["model"]:
            num_patches = config["model"]["num_patches"]

        img_width = 4096.0 / num_patches
        img_height = 4096.0 / num_patches
        cell_w = img_width / grid_w
        cell_h = img_height / grid_h

        box_width, box_height = config_tools.get_box_size(config)

        anchor_width = 1.0 * box_width / cell_w
        anchor_height = 1.0 * box_height / cell_h
        anchors = [anchor_width, anchor_height]
    elif anchors is None:
        num_patches = 1
        if "num_patches" in config["model"]:
            num_patches = config["model"]["num_patches"]
        img_width = 4096.0 / num_patches
        img_height = 4096.0 / num_patches
        cell_w = img_width / grid_w
        cell_h = img_height / grid_h
        anchors = np.array(config["model"]["anchors"], dtype=float)
        anchors[::2] = anchors[::2] / cell_w
        anchors[1::2] = anchors[1::2] / cell_h

    return anchors


def prepare_datasets(
        path_config, path_weights, path_boxfiles, path_runfile, path_images
):
    datasets = {}

    if path_runfile:
        # Read run file
        with open(path_runfile) as runfile_buffer:
            runfile = json.load(runfile_buffer)
        images = runfile["run"]["valid_images"]
        annot = runfile["run"]["valid_annot"]

        for i in range(len(images)):
            imgpath = images[i]
            boxpath = annot[i]

            if os.stat(boxpath).st_size == 0:
                continue
            last_folder = os.path.basename(os.path.dirname(imgpath))
            if last_folder in datasets:
                box_files = datasets[last_folder]["boxes"]
                img_files = datasets[last_folder]["images"]
            else:
                box_files = []
                img_files = []
                datasets[last_folder] = {}
                datasets[last_folder]["boxes"] = box_files
                datasets[last_folder]["images"] = img_files
            box_files.append(boxpath)
            img_files.append(imgpath)
    else:
        # Scan all box filenames
        for root, directories, filenames in os.walk(path_boxfiles):
            if len(filenames) == 0:
                continue
            last_folder = os.path.basename(root)
            if last_folder.startswith(("empty", ".")):
                # Ignore empty grids
                continue

            box_files = []
            for filename in filenames:
                if filename.endswith(("box")) and not filename.startswith("."):
                    box_files.append(os.path.join(root, filename))

            if len(box_files) > 1:
                if last_folder in datasets:
                    datasets[last_folder]["boxes"].append(box_files)
                else:
                    dataset = {"boxes": box_files}
                    datasets[last_folder] = dataset

        # Find all images
        img_files = []
        for root, directories, filenames in os.walk(path_images):
            last_folder = os.path.basename(root)
            if last_folder.startswith("empty"):
                # Ignore empty grids
                continue

            for filename in filenames:
                if filename.endswith(
                        ("jpg", "png", "tiff", "tif", "mrc")
                ) and not filename.startswith("."):
                    img_files.append(os.path.join(root, filename))

        # Find to boxifles corresponding images
        for foldername, dataset in datasets.items():
            boxfiles = dataset["boxes"]
            imgfiles = []
            for boxfile in boxfiles:
                filename = os.path.splitext(os.path.basename(boxfile))[0]
                lastfolder = os.path.basename(os.path.dirname(boxfile))
                search_path = os.path.join(lastfolder, filename)
                corresponding_img_path = difflib.get_close_matches(
                    search_path, img_files, n=1, cutoff=0
                )[0]
                print("Assign", boxfile, "to", corresponding_img_path)
                imgfiles.append(corresponding_img_path)
            dataset["images"] = imgfiles

    # Read config
    with open(path_config) as config_buffer:
        config = json.load(config_buffer)

    # Configure model
    ###############################
    #   Make the model
    ###############################
    depth = 1
    num_gpus = 1

    backend_weights = None

    anchors = get_anchors(config, path_weights)

    if "backend_weights" in config["model"]:
        backend_weights = config["model"]["backend_weights"]

    yolo = YOLO(
        architecture=config["model"]["architecture"],
        input_size=config["model"]["input_size"],
        input_depth=depth,
        labels=["particle"],
        max_box_per_image=config["model"]["max_box_per_image"],
        anchors=anchors,
        backend_weights=backend_weights,
    )

    ###############################
    #   Load trained weights
    ###############################
    try:
        yolo.load_weights(path_weights)
    except ValueError:

        # print(traceback.format_exc())
        import sys
        sys.exit(
            "Seems that the architecture in your config (-c parameter) "
            "does not fit the model weights (-w parameter)"
        )
    # USE MULTIGPU
    if num_gpus > 1:
        parallel_model = multi_gpu_model(yolo.model, gpus=num_gpus)
        yolo.model = parallel_model

    # Get picks
    for foldername, dataset in datasets.items():
        # Get image dimensions of current dataset
        width, height = imagereader.read_width_height(dataset["images"][0])

        # Get box size
        one_box_file = read_box(dataset["boxes"][0], height)
        box_size = one_box_file[0]["w"]
        # Adapt the anchor size in config file.
        config["model"]["particle_diameter"] = box_size
        # if is_single_anchor:
        #     anchor_width = 1.0 * box_size / cell_w
        #     anchor_height = 1.0 * box_size / cell_h
        #     anchors = [anchor_width, anchor_height]
        #     yolo.anchors = anchors

        num_patches = config_tools.get_number_patches(config)

        # Get overlap patches
        overlap_patches = 0
        if "overlap_patches" in config["model"]:
            overlap_patches = int(config["model"]["overlap_patches"])
        elif not len(config["model"]["anchors"]) > 2:
            overlap_patches = config["model"]["anchors"][0]

        # Predict
        predict_results = predict.do_prediction(
            config_path=path_config,
            weights_path=path_weights,
            num_patches=num_patches,
            input_path=dataset["images"],
            obj_threshold=0.0,
            config_pre=config,
            overlap=overlap_patches,
            yolo=yolo,
        )

        # Rescale to image dimensions
        for boxes_in_micrograph in predict_results:
            image_width = boxes_in_micrograph["img_width"]
            image_height = boxes_in_micrograph["img_height"]
            for box in boxes_in_micrograph["boxes"]:
                x_ll = int((box.x - box.w / 2) * image_height)  # lower left
                y_ll = int(
                    image_width - box.y * image_width - box.h / 2 * image_width
                )  # lower right
                boxheight_in_pxl = int(box.h * image_width)
                boxwidth_in_pxl = int(box.w * image_height)

                delta_boxsize_height = box_size - boxheight_in_pxl
                delta_boxsize_width = box_size - boxwidth_in_pxl
                x_ll = x_ll - delta_boxsize_height / 2
                y_ll = y_ll - delta_boxsize_width / 2
                boxsize_in_pxl = box_size

                box.x = x_ll + boxsize_in_pxl / 2
                box.y = y_ll + boxsize_in_pxl / 2
                box.w = boxsize_in_pxl
                box.h = boxsize_in_pxl

        dataset["picks"] = predict_results

    return datasets


def calculate_all_statistics(datasets):
    arg_list = []

    for foldername, dataset in datasets.items():
        arg_list.append((foldername, dataset))

    pool = multiprocessing.Pool()
    result_dicts = pool.starmap(calculate_statics_dataset, arg_list)
    pool.close()
    pool.join()

    k = 0
    for foldername, dataset in datasets.items():
        dataset["results"] = result_dicts[k]
        """
        result_rows = zip(result_dicts[k]["RECALL_LIST"], result_dicts[k]["PRECISION_LIST"])

        import csv
        with open("recall_precision_"+foldername+".csv", "w") as f:
            writer = csv.writer(f, delimiter=';')
            for row in result_rows:

                writer.writerow(row)

        f.close()
        """
        k = k + 1


def calculate_statics_dataset(foldername, dataset):
    arange = np.arange(start=0, stop=1.001, step=0.001)
    result_dict = {}
    maxf1 = 0.0
    maxf1_thresh = 0.0
    maxf2_thresh = 0.0
    maxf2 = 0.0
    recall_list = []
    precision_list = []
    f1_list = []
    f2_list = []
    sum_picked_list = []

    for thresh in arange:
        thresh = round(thresh, 3)
        res = get_precision_recall_f1(dataset, thresh)
        if res is not None:
            result_dict[str(thresh)] = res
            precision = result_dict[str(thresh)][1]
            recall = result_dict[str(thresh)][2]
            sum_picked = result_dict[str(thresh)][4]
            f1 = result_dict[str(thresh)][0]
            f2 = result_dict[str(thresh)][5]
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            f2_list.append(f2)
            sum_picked_list.append(sum_picked)
            # if len(recall_list) > 1:
            #    area.append((recall_list[-2] - recall_list[-1]) * precision_list[-1])

            if f1 > maxf1:
                maxf1 = f1
                maxf1_thresh = thresh
            if f2 > maxf2:
                maxf2 = f2
                maxf2_thresh = thresh

    # AUC Calculation
    precision_array = np.array(precision_list)
    recall_array = np.array(recall_list)

    # Interpolate first point
    recall_array = np.insert(recall_array, len(recall_array), 0)
    precision_array = np.insert(
        precision_array, len(precision_array), precision_array[len(precision_array) - 1]
    )

    # Sort
    sorted_index = np.argsort(recall_array)[::-1]
    precision_array = precision_array[sorted_index]
    recall_array = recall_array[sorted_index]

    area = 0
    for i in range(1, len(recall_array)):
        val = (recall_array[i - 1] - recall_array[i]) * precision_array[i]
        area = area + val

    result_dict["AUC"] = area
    result_dict["MAX_F1_THRESH"] = maxf1_thresh
    result_dict["MAX_F2_THRESH"] = maxf2_thresh
    result_dict["THRESH_LIST"] = arange
    result_dict["RECALL_LIST"] = recall_array.tolist()
    result_dict["PRECISION_LIST"] = precision_array.tolist()
    result_dict["F1_LIST"] = f1_list
    result_dict["F2_LIST"] = f2_list
    result_dict["PICKED_LIST"] = sum_picked_list
    return result_dict


def get_precision_recall_f1(dataset, confidence_threshold):
    results_per_threshold = {}

    f1_sum = 0
    f2_sum = 0
    precision_sum = 0
    recall_sum = 0
    mean_iou_sum = 0
    iou_tresh = 0.6
    picks = dataset["picks"]

    boxes = dataset["boxes"]
    images = dataset["images"]
    sum_picked = 0
    images_with_particles = 0
    for boxes_in_mic in picks:
        TP = 0
        MEAN_IOU = 0
        gd_box_remove = []
        picked_boxes_index_remove = []

        picked_boxes = boxes_in_mic["boxes"]
        img_path = boxes_in_mic["img_path"]
        img_name_without_suffix = os.path.splitext(os.path.basename(img_path))[0]
        # ground_truth_boxes_path = difflib.get_close_matches(img_name_without_suffix, boxes, n=1, cutoff=0)[0]

        cand_list = [
            i
            for i in boxes
            if os.path.splitext(os.path.basename(i))[0] in img_name_without_suffix
        ]
        try:
            cand_list_no_fileextension = list(map(os.path.basename, cand_list))
            ground_truth_boxes_path = difflib.get_close_matches(
                img_name_without_suffix, cand_list_no_fileextension, n=1, cutoff=0
            )[0]
            ground_truth_boxes_path = cand_list[
                cand_list_no_fileextension.index(ground_truth_boxes_path)
            ]
        except IndexError as e:
            print("Cannot find corresponding boxfile for ", img_path)
            raise e

        # Read ground truth boxes:
        _, height = imagereader.read_width_height(img_path)
        ground_truth_boxes = read_box(ground_truth_boxes_path, height)
        picked_boxes_filtered = [
            x for x in picked_boxes if x.get_score() > confidence_threshold
        ]
        sum_picked = sum_picked + len(picked_boxes_filtered)
        # Ground Truth boxes to array
        gd_boxes_data = np.empty(shape=(len(ground_truth_boxes), 5))
        for i, gdbox in enumerate(ground_truth_boxes):
            gd_boxes_data[i, 0] = gdbox["x"]
            gd_boxes_data[i, 1] = gdbox["y"]
            gd_boxes_data[i, 2] = gdbox["w"]
            gd_boxes_data[i, 3] = gdbox["h"]
            gd_boxes_data[i, 4] = i

        # Picked boxes to array
        picked_boxes_data = np.empty(shape=(len(picked_boxes_filtered), 5))
        for i, pbox in enumerate(picked_boxes_filtered):
            picked_boxes_data[i, 0] = pbox.x
            picked_boxes_data[i, 1] = pbox.y
            picked_boxes_data[i, 2] = pbox.w
            picked_boxes_data[i, 3] = pbox.h
            picked_boxes_data[i, 4] = i
        # print("GD LEN",len(gd_boxes_data))
        # print("PICKED LEN", len(picked_boxes_data), "Tresh",confidence_threshold, "img",img_path)
        for i, gd_box in enumerate(ground_truth_boxes):

            # Run fast methods here
            if len(picked_boxes_data) > 0:
                box_i = gd_boxes_data[i, :]
                boxes_i_rep = np.array([box_i] * len(picked_boxes_data))
                ious = bbox_iou_vec(boxes_i_rep, picked_boxes_data)

                # Delete that
                # ious = calc_ious(gd_box,picked_boxes_filtered)
                max_iou = 0
                p_box_index = 0
                if len(ious) > 0:
                    p_box_index = np.argmax(ious)
                    max_iou = ious[p_box_index]

                # print max_iou
                if max_iou > iou_tresh:
                    gd_box_remove.append(gd_box)
                    # p_box_index = ious.index(max_iou)
                    picked_boxes_index_remove.append(p_box_index)
                    TP = TP + 1
                    MEAN_IOU += max_iou
        if len(picked_boxes_filtered) > 0:
            FP = len(picked_boxes_filtered) - len(picked_boxes_index_remove)
            FN = len(ground_truth_boxes) - len(gd_box_remove)
            if (TP + FP) == 0:
                precision = 0
            else:
                precision = 1.0 * TP / (TP + FP)

            recall = 1.0 * TP / (TP + FN)
            if (precision + recall) == 0:
                F1 = 0
                F2 = 0
            else:
                F1 = 2.0 * precision * recall / (precision + recall)
                F2 = 5.0 * precision * recall / (4 * precision + recall)
            f1_sum += F1
            f2_sum += F2
            precision_sum += precision
            recall_sum += recall
            mean_iou_sum += MEAN_IOU / (TP + 0.001)

            images_with_particles = images_with_particles + 1

    if images_with_particles == 0:
        return None

    # --------------------------------------------#
    f1_avg = f1_sum / images_with_particles
    f2_avg = f2_sum / images_with_particles
    precision_avg = precision_sum / images_with_particles
    recall_avg = recall_sum / images_with_particles
    mean_iou_sum = mean_iou_sum / images_with_particles
    return (f1_avg, precision_avg, recall_avg, mean_iou_sum, sum_picked, f2_avg)


def calc_stats_single_mic(boxes_in_mic, boxes, iou_tresh, confidence_threshold):
    TP = 0
    MEAN_IOU = 0
    gd_box_remove = []
    picked_boxes_index_remove = []

    picked_boxes = boxes_in_mic["boxes"]
    img_path = boxes_in_mic["img_path"]
    img_name_without_suffix = os.path.splitext(os.path.basename(img_path))[0]
    # ground_truth_boxes_path = difflib.get_close_matches(img_name_without_suffix, boxes, n=1, cutoff=0)[0]

    cand_list = [
        i
        for i in boxes
        if os.path.splitext(os.path.basename(i))[0] in img_name_without_suffix
    ]
    try:
        cand_list_no_fileextension = list(map(os.path.basename, cand_list))
        ground_truth_boxes_path = difflib.get_close_matches(
            img_name_without_suffix, cand_list_no_fileextension, n=1, cutoff=0
        )[0]
        ground_truth_boxes_path = cand_list[
            cand_list_no_fileextension.index(ground_truth_boxes_path)
        ]
    except IndexError as e:
        print("Cannot find corresponding boxfile for ", img_path)
        raise e

    # Read ground truth boxes:
    _, height = imagereader.read_width_height(img_path)
    ground_truth_boxes = read_box(ground_truth_boxes_path, height)
    picked_boxes_filtered = [x for x in picked_boxes if x.c > confidence_threshold]

    # Ground Truth boxes to array
    gd_boxes_data = np.empty(shape=(len(ground_truth_boxes), 5))
    for i, gdbox in enumerate(ground_truth_boxes):
        gd_boxes_data[i, 0] = gdbox["x"]
        gd_boxes_data[i, 1] = gdbox["y"]
        gd_boxes_data[i, 2] = gdbox["w"]
        gd_boxes_data[i, 3] = gdbox["h"]
        gd_boxes_data[i, 4] = i

    # Picked boxes to array
    picked_boxes_data = np.empty(shape=(len(picked_boxes_filtered), 5))
    for i, pbox in enumerate(picked_boxes_filtered):
        picked_boxes_data[i, 0] = pbox.x
        picked_boxes_data[i, 1] = pbox.y
        picked_boxes_data[i, 2] = pbox.w
        picked_boxes_data[i, 3] = pbox.h
        picked_boxes_data[i, 4] = i
    # print("GD LEN",len(gd_boxes_data))
    # print("PICKED LEN", len(picked_boxes_data))
    for i, gd_box in enumerate(ground_truth_boxes):

        # Run fast methods here
        if len(picked_boxes_data) > 0:
            box_i = gd_boxes_data[i, :]
            boxes_i_rep = np.array([box_i] * len(picked_boxes_data))
            ious = bbox_iou_vec(boxes_i_rep, picked_boxes_data)

            # Delete that
            # ious = calc_ious(gd_box,picked_boxes_filtered)
            max_iou = 0
            p_box_index = 0
            if len(ious) > 0:
                p_box_index = np.argmax(ious)
                max_iou = ious[p_box_index]

            # print max_iou
            if max_iou > iou_tresh:
                gd_box_remove.append(gd_box)
                # p_box_index = ious.index(max_iou)
                picked_boxes_index_remove.append(p_box_index)
                TP = TP + 1
                MEAN_IOU += max_iou
    FP = len(picked_boxes_filtered) - len(picked_boxes_index_remove)
    FN = len(ground_truth_boxes) - len(gd_box_remove)

    if (TP + FP) == 0:
        precision = 0
    else:
        precision = 1.0 * TP / (TP + FP)

    recall = 1.0 * TP / (TP + FN)
    if (precision + recall) == 0:
        F1 = 0
    else:
        F1 = 2.0 * precision * recall / (precision + recall)
    return (F1, precision, recall, MEAN_IOU / (TP + 0.001))


def calc_ious(ground_truth_box, picked_boxes):
    return [bbox_iou(ground_truth_box, p_box) for p_box in picked_boxes]


def read_box(path, image_height):
    boxes = []
    # with open(path, 'r') as boxfile:
    boxreader = np.atleast_2d(
        np.genfromtxt(path)
    )  # csv.reader(boxfile, delimiter='\t',
    #          quotechar='|', quoting=csv.QUOTE_NONE)

    try:
        for box in boxreader:
            box_d = {}
            width = int(float(box[2]))
            height = int(float(box[3]))

            box_d["x"] = int(float(box[0])) + width / 2
            box_d["y"] = int(float(box[1])) + height / 2
            box_d["w"] = width
            box_d["h"] = height
            box_d["filename"] = path
            boxes.append(box_d)
    except Exception as e:
        print("ReadBox error. Path:", path)
        print(e)

    return boxes


def read_boxes(box_dir, image_height):
    onlyfiles = [
        f for f in os.listdir(box_dir) if os.path.isfile(os.path.join(box_dir, f))
    ]
    boxes = []
    for file in onlyfiles:
        if file.endswith(".box"):
            path = os.path.join(box_dir, file)
            box = read_box(path, image_height)
            if box is not None:
                boxes.append(box)
    return boxes


def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou_vec(boxesA, boxesB):
    x1_min = boxesA[:, 0] - boxesA[:, 2] / 2
    x1_max = boxesA[:, 0] + boxesA[:, 2] / 2
    y1_min = boxesA[:, 1] - boxesA[:, 3] / 2
    y1_max = boxesA[:, 1] + boxesA[:, 3] / 2

    x2_min = boxesB[:, 0] - boxesB[:, 2] / 2
    x2_max = boxesB[:, 0] + boxesB[:, 2] / 2
    y2_min = boxesB[:, 1] - boxesB[:, 3] / 2
    y2_max = boxesB[:, 1] + boxesB[:, 3] / 2
    intersect_w = interval_overlap_vec(x1_min, x1_max, x2_min, x2_max)
    intersect_h = interval_overlap_vec(y1_min, y1_max, y2_min, y2_max)
    intersect = intersect_w * intersect_h
    union = boxesA[:, 2] * boxesA[:, 3] + boxesB[:, 2] * boxesB[:, 3] - intersect
    return intersect / union


def interval_overlap_vec(x1_min, x1_max, x2_min, x2_max):
    intersect = np.zeros(shape=(len(x1_min)))
    condA = x2_min < x1_min
    condB = condA & (x2_max >= x1_min)
    intersect[condB] = np.minimum(x1_max[condB], x2_max[condB]) - x1_min[condB]
    condC = ~condA & (x1_max >= x2_min)
    intersect[condC] = np.minimum(x1_max[condC], x2_max[condC]) - x2_min[condC]

    return intersect


def bbox_iou(box1, box2):
    x1_min = box1["x"] - box1["w"] / 2
    x1_max = box1["x"] + box1["w"] / 2
    y1_min = box1["y"] - box1["h"] / 2
    y1_max = box1["y"] + box1["h"] / 2

    x2_min = box2.x - box2.w / 2
    x2_max = box2.x + box2.w / 2
    y2_min = box2.y - box2.h / 2
    y2_max = box2.y + box2.h / 2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = box1["w"] * box1["h"] + box2.w * box2.h - intersect

    return float(intersect) / union


class picking_result_folder:
    def __init__(self, foldername):
        self.foldername = foldername
        self.boxes = {}

    def add_boxes(self, image_name, boxes_dict):
        self.boxes[image_name] = boxes_dict


if __name__ == "__main__":
    _main_()

