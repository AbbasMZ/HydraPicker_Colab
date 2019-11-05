import os, sys
import numpy as np


def create_results_folder(directory):
    # Creates a results folder if one doesn't already exist.

    if not os.path.isdir(directory):
        os.makedirs(directory)


def read_results_file_array(directory):
    # Opens results file and reads contents into a list.

    # read the results file and split the rows into a list
    try:
        results_file = open(directory + "/results.csv","r+")
    except FileNotFoundError:
        print("Error: results file missing!")
        sys.exit(3)

    results_list_csv = results_file.read().split("\n")
    results_file.close()

    # split every row into a result entry by applying read_result_entry
    results_map = map(read_result_entry_array, results_list_csv)
    # filter out first (header) and last (empty) lines
    results_list = list(filter(lambda x: x != None, results_map))

    return np.asarray(results_list).astype(float)


def read_results_file_ctffind4(directory, id_number, test_number):
    # Opens results file and reads contents into a list.

    # read the results file and split the rows into a list
    try:
        results_file = open(directory + "/results.csv","r+")
    except FileNotFoundError:
        print("Error: results file missing!")
        sys.exit(3)

    results_list_csv = results_file.read().split("\n")
    results_file.close()

    # split every row into a result entry by applying read_result_entry
    results_map = map(read_result_entry_array, results_list_csv)
    # filter out first (header) and last (empty) lines
    results_list = list(filter(lambda x: x != None, results_map))

    return np.asarray(results_list).astype(float)


def read_result_entry_array(result_text):
    # Parses a result entry from comma-separated to array form."""

    # split comma-separated values
    result_list = result_text.split(",")

    # handle invalid lines
    if len(result_list) > 1 and result_list[0] != "df1":
        # result_list.astype(np.float)
        return result_list
    else:
        return None


def read_result_entry_ctffind4(result_text):
    # Parses a result entry from comma-separated to array form.

    # split comma-separated values
    result_list = result_text.split(",")

    # handle invalid lines
    if len(result_list) > 1 and result_list[0] != "#":
        return result_list
    else:
        return None


def write_results_file(directory, results_list, type):
    # Creates a new or empties the existing results file and fills it with the list of results.

    # (Re)create empty results file
    create_results_folder(directory)

    if type == 0:

        results_file = open(directory + "/results.csv", "w+")

        # Write results file
        results_file.write("method,d_avg,d_dif,angast,d_avg_t,d_dif_t,angast_t,d_avg_e,d_dif_e,angast_e\n")
        for result in results_list:
            results_file.write(
                str(result['method']) + ","
                + str(result['d_avg']) + ","
                + str(result['d_dif']) + ","
                + str(result['angast']) + ","
                + str(result['d_avg_t']) + ","
                + str(result['d_dif_t']) + ","
                + str(result['angast_t']) + ","
                + str(result['d_avg_e']) + ","
                + str(result['d_dif_e']) + ","
                + str(result['angast_e']) + "\n")

    elif type == 2 or type == 3 or type == 4:
        if type == 2:
            results_file = open(directory + "/results2.csv", "w+")
        elif type == 3:
            results_file = open(directory + "/results3.csv", "w+")
        elif type == 4:
            results_file = open(directory + "/results4.csv", "w+")
        else:
            print("Error: results type not correct!")
            sys.exit(3)

        # Write results file
        results_file.write("d_avg_e_mean, d_avg_e_std, d_dif_e_mean, d_dif_e_std, d_ang_e_mean, angast_e_std\n")
        for result in results_list:
            results_file.write(
                str(result['d_avg_e_mean']) + ","
                + str(result['d_avg_e_std']) + ","
                + str(result['d_dif_e_mean']) + ","
                + str(result['d_dif_e_std']) + ","
                + str(result['angast_e_mean']) + ","
                + str(result['angast_e_std']) + "\n")
    else:
        print("Error: results type not correct!")
        sys.exit(3)

    # Close the file
    results_file.close()


def read_output_ctffind4(case_num, id2, path='', filename=''):
    # read the results file and split the rows into a list
    try:
        if filename == '':
            filename = "ctffind4_"+str(case_num)+"_"+str(id2)+".txt"

        results_file = open(path+filename, "r+")

    except FileNotFoundError:
        print("Error: results file missing!")
        sys.exit(3)

    results_list = results_file.read().split("\n")
    results_file.close()

    results_row = results_list[5].split(" ")

    df1 = []
    df2 = []
    ang = []
    df1.append(np.asarray(results_row[1]).astype(float))
    df2.append(np.asarray(results_row[2]).astype(float))
    ang.append(np.asarray(results_row[3]).astype(float))
    return [df1, df2, ang]


def read_output_ctffind3(case_num, id2):
    # read the results file and split the rows into a list
    try:
        results_file = open("ctffind3_output_"+str(case_num)+"_"+str(id2)+".txt", "r+")
    except FileNotFoundError:
        print("Error: results file missing!")
        sys.exit(3)

    results_list = results_file.read().split("\n")
    results_file.close()

#    results_row = results_list[5].split(" ")

    for i in range(0,len(results_list)):
        results_row = results_list[i].split(" ")
        if len(results_row) > 1:
            if results_row[1] == "REFINING":
                results_row = results_list[i+4].split(" ")
                break
    df1 = []
    df2 = []
    ang = []
    flag = 0;
    for i in range(0,len(results_row)):
        if results_row[i] != "" and flag == 0:
            # df1 = np.asarray(results_row[i]).astype(float)
            df1.append(np.asarray(results_row[i]).astype(float))
            flag = 1
            continue
        if results_row[i] != "" and flag == 1:
            # df2 = np.asarray(results_row[i]).astype(float)
            df2.append(np.asarray(results_row[i]).astype(float))
            flag = 2
            continue
        if results_row[i] != "" and flag == 2:
            # ang = np.asarray(results_row[i]).astype(float)
            ang.append(np.asarray(results_row[i]).astype(float))
            break

    return [df1, df2, ang]


def read_output_gctf(id1, id2, type):
    # read the results file and split the rows into a list
    try:
        # results_file = open("ctffind3_t"+str(case_num)+"_1.txt", "r+")
        results_file = open("imgdata_"+str(id1)+"_"+str(id2)+"_gctf.log", "r+")
    except FileNotFoundError:
        print("Error: results file missing!")
        sys.exit(3)

    results_list = results_file.read().split("\n")
    results_file.close()

    if type == "global":
        for counter1 in range(0, len(results_list)):
            if len(results_list[counter1]) > 1:
                if results_list[counter1] == '**************************************   LAST CYCLE    ************************************************************ *':
                    results_row = results_list[counter1 + 3].split(" ")
                    break
        df1 = []
        df2 = []
        ang = []
        flag = 0
        for i in range(0, len(results_row)):
            if results_row[i] != "":
                if flag == 0:
                    # df1 = np.asarray(results_row[i]).astype(float)
                    df1.append(np.asarray(results_row[i]).astype(float))
                    flag = 1
                    continue
                elif flag == 1:
                    # df2 = np.asarray(results_row[i]).astype(float)
                    df2.append(np.asarray(results_row[i]).astype(float))
                    flag = 2
                    continue
                elif flag == 2:
                    # ang = np.asarray(results_row[i]).astype(float)
                    ang.append(np.asarray(results_row[i]).astype(float))
                    break

        return [df1, df2, ang]
    elif type == "local":
        for counter1 in range(0, len(results_list)):
            if len(results_list[counter1]) > 1:
                if results_list[
                    counter1] == '**************************************   LAST CYCLE    ************************************************************ *':
                    break

        df1 = []
        df2 = []
        ang = []
        for counter2 in range(counter1 + 10, len(results_list) - 2):
            results_row = results_list[counter2].split(" ")
            flag = 0
            for i in range(0, len(results_row)):
                if results_row[i] != "":
                    if flag == 0:
                        flag = 1
                        continue
                    elif flag == 1:
                        flag = 2
                        continue
                    elif flag == 2:
                        df1.append(np.asarray(results_row[i]).astype(float))
                        flag = 3
                        continue
                    elif flag == 3:
                        df2.append(np.asarray(results_row[i]).astype(float))
                        flag = 4
                        continue
                    elif flag == 4:
                        ang.append(np.asarray(results_row[i]).astype(float))
                        break

        return [df1, df2, ang]
    else:
        print("Error: gctf type not correct!")
        sys.exit(3)


def write_local_box_file(local_points_list, local_points_number, local_window_size, id_number, micrograph_counter):
    # Creates a new or empties the existing local box file and fills it with the list of points.

    # (Re)create empty points file
    box_file = open("imgdata_" + str(id_number) + "_" + str(micrograph_counter) + "_automatch.box", "w+")

    # Write box file
    for i in range(0, local_points_number):
        box_file.write(str(local_points_list[i][0]) + "    " + str(local_points_list[i][1]) + "    " + str(local_window_size) + "     " + str(local_window_size) + "     " + "-3\n")

    # Close the file
    box_file.close()


def write_local_indexes_file(local_points_indexes, local_points_number, id_number, micrograph_counter):
    # Creates a new or empties the existing local indexes file and fills it with the list of indexes to defocus.

    # (Re)create empty points file
    indexes_file = open("local_indexes_" + str(id_number) + "_" + str(micrograph_counter) + ".txt", "w+")

    # Write local indexes file
    for i in range(0, local_points_number):
        indexes_file.write(str(local_points_indexes[i]) + "\n")

    # Close the file
    indexes_file.close()


def write_ctffind4_input_files(index1, index2=1, pixel=2.8, volt=200, s_aberration=2.0, amp_contrast=0.07, amp_spectrum=512, min_res=30, max_res=6, min_def=5000, max_def=50000, def_step=500, known_astig='no', slow='yes', restraint='no', phase_shift='no', expert='no', path='', mrc_filename='', input_filename=''):
    # Creates required input files for ctffind4.

    if input_filename == '':
        input_filename = "ctffind4_input_" + str(index1) + "_" + str(index2) + ".txt"

    input_path = os.path.join(path, input_filename)
    ctffind4_input_file = open(input_path, "w+")

    if mrc_filename == '':
        mrc_filename = "imgdata_" + str(index1) + "_" + str(index2)
    # "imgdata_" + str(index1) + "_" + str(index2) + ".mrc\n"
    ctffind4_input_file.write(
        mrc_filename + ".mrc\n"
        + "ctffind4_" + str(index1) + "_" + str(index2) + ".mrc\n"
        + str(pixel) + "\n"
        + str(volt) + "\n"
        + str(s_aberration) + "\n"
        + str(amp_contrast) + "\n"
        + str(amp_spectrum) + "\n"
        + str(min_res) + "\n"
        + str(max_res) + "\n"
        + str(min_def) + "\n"
        + str(max_def) + "\n"
        + str(def_step) + "\n"
        + known_astig + "\n"
        + slow + "\n"
        + restraint + "\n"
        + phase_shift + "\n"
        + expert)

    # Close the file
    ctffind4_input_file.close()


def write_ctffind3_input_files(index1, index2=1, cs=2, ht=200, ampcnst=0.07, xmag=60000, dstep=16.8, box=512, resmin=30, resmax=6, dfmin=5000, dfmax=50000, fstep=500, dast=1000):
    # Creates required input files for ctffind3.

    ctffind3_input_file = open("ctffind3_input_" + str(index1) + "_" + str(index2) + ".txt", "w+")

    ctffind3_input_file.write(
        "imgdata_" + str(index1) + "_" + str(index2) + ".mrc\n"
        + "ctffind3_" + str(index1) + "_" + str(index2) + ".mrc\n"
        + str(cs) + ", "
        + str(ht) + ", "
        + str(ampcnst) + ", "
        + str(xmag) + ", "
        + str(dstep) + "\n"
        + str(box) + ", "
        + str(resmin) + ", "
        + str(resmax) + ", "
        + str(dfmin) + ", "
        + str(dfmax) + ", "
        + str(fstep) + ", "
        + str(dast))

    # Close the file
    ctffind3_input_file.close()


def read_true_defocus(case_num, path=""):
    # read the true defocus file
    dpath = path + "defocus_"+str(case_num)+"_1.txt"
    try:
        results_file = open(dpath, "r+")

    except FileNotFoundError:
        print("Error: results file missing!")
        sys.exit(3)

    results_list = results_file.read().split("\n")
    results_file.close()
    results_row = results_list[0].split(" ")
    for i in range(0, len(results_row)):
        if len(results_row[i]) > 1:
            if results_row[i] == "'v_d_avg':":
                df_avg = results_row[i + 1].split(",")[0]
                continue
            if results_row[i] == "'v_d_dif':":
                df_dif = results_row[i + 1].split(",")[0]
                continue
            if results_row[i] == "'angast':":
                angast = results_row[i + 1].split(",")[0]
                continue
    return [np.asarray(df_avg), np.asarray(df_dif), np.asarray(angast)]