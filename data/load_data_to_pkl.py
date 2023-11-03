import warnings
import sys
import getopt
import pandas as pd
import os
import time
import random
import numpy as np
import seaborn as sns

from tqdm import tqdm
from os import listdir
from skimage.transform import resize
from skimage.io import imread


sns.set()
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


FILE_DIR = os.listdir("/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/")
BASE_PATH = "/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/"
PICKLE_PATH = "/scratch/owebb1/"
FOLDER = os.listdir(BASE_PATH)
SIZE = 10
FILE_NUM = 1


# DEFINITION:: Will extract x and y coordinates from the path to file
def extract_coords(df):
    coord = df.path.str.rsplit("_", n=4, expand=True)
    coord = coord.drop([0, 1, 4], axis=1)
    coord = coord.rename({2: "x", 3: "y"}, axis=1)
    coord.loc[:, "x"] = (
        coord.loc[:, "x"].str.replace("x", "", case=False).astype(np.int)
    )
    coord.loc[:, "y"] = (
        coord.loc[:, "y"].str.replace("y", "", case=False).astype(np.int)
    )
    df.loc[:, "x"] = coord.x.values
    df.loc[:, "y"] = coord.y.values
    return df


# DEFINITION: Creates a new dataframe and puts image data in it
#     Columns = target, path, x, y
def get_cancer_dataframe(patient_id, cancer_id):
    path = BASE_PATH + patient_id + "/" + cancer_id
    files = listdir(path)
    dataframe = pd.DataFrame(files, columns=["filename"])
    path_names = path + "/" + dataframe.filename.values
    dataframe = dataframe.filename.str.rsplit("_", n=4, expand=True)
    dataframe.loc[:, "target"] = np.int(cancer_id)
    dataframe.loc[:, "path"] = path_names
    dataframe = dataframe.drop([0, 1, 4], axis=1)
    dataframe = dataframe.rename({2: "x", 3: "y"}, axis=1)
    dataframe.loc[:, "x"] = (
        dataframe.loc[:, "x"].str.replace("x", "", case=False).astype(np.int)
    )
    dataframe.loc[:, "y"] = (
        dataframe.loc[:, "y"].str.replace("y", "", case=False).astype(np.int)
    )
    return dataframe


# DEFINITION: Simply calls for certain patient for positive and negative
def get_patient_dataframe(patient_id):
    df_0 = get_cancer_dataframe(patient_id, "0")
    df_1 = get_cancer_dataframe(patient_id, "1")
    patient_df = df_0.append(df_1)
    return patient_df


# DEFINITION: Load the image data into the dataframes and save it as a pkl file
def original_save_as_pickle(filename):
    print("Begining to Load Original Dataframe")
    target_arr = []
    # datadir='/Users/owenwebb/E90_data/archive-2/IDC_regular_ps50_idx5'
    patient_id_arr = []
    path_arr = []
    for n in range(len(FOLDER)):
        # for n in range(len(FOLDER)):
        patient_id = FOLDER[n]
        patient_path = BASE_PATH + patient_id
        for c in [0, 1]:
            class_path = patient_path + "/" + str(c) + "/"
            subfiles = os.listdir(class_path)
            for img in range(len(subfiles)):
                image_path = subfiles[img]
                complete_path = os.path.join(class_path, image_path)
                target_arr.append(c)
                patient_id_arr.append(patient_id)
                path_arr.append(complete_path)

    target = np.array(target_arr)
    patient_id_np = np.array(patient_id_arr)
    path_arr_np = np.array(path_arr)
    data = pd.DataFrame()
    data["target"] = target
    data["patient_id"] = patient_id_np
    data["path"] = path_arr_np

    data.to_pickle(filename)

    print("Done Loading Original Dataframe")
    return data


# DEFINITION : Load the image data into the dataframes and save it as a pkl file
def original_save_as_pickle_small(small_img_pkl, size):
    print("Beginning to Load Non Image Dataframe Subset")
    target_arr = []
    patient_id_arr = []
    path_arr = []
    for _ in range(size):
        # patient_id = FOLDER[n]
        patient_id = random.choice(FOLDER)
        patient_path = BASE_PATH + patient_id
        for c in [0, 1]:
            class_path = patient_path + "/" + str(c) + "/"
            subfiles = os.listdir(class_path)
            for img in range(len(subfiles)):
                image_path = subfiles[img]
                complete_path = os.path.join(class_path, image_path)
                target_arr.append(c)
                patient_id_arr.append(patient_id)
                path_arr.append(complete_path)

    target = np.array(target_arr)
    patient_id_np = np.array(patient_id_arr, dtype=int)
    path_arr_np = np.array(path_arr)
    data = pd.DataFrame()
    data["target"] = target
    data["patient_id"] = patient_id_np
    data["path"] = path_arr_np

    data.to_pickle(small_img_pkl)

    print("Done Loading Non Image Dataframe Subset")

    return data


# DEFINITION: flatten the image and put it into a large dataframe
def flatten_image_and_new_df(data, filename):
    print("Beginning to Load Image Dataframe")
    start = time.time()
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        img_path = row["path"]
        img_array = imread(img_path)
        img_resized = resize(img_array, (50, 50, 3))
        img_flatten = np.array(img_resized.flatten(), dtype=float).reshape(
            1, 7500
        )
        if i == 0:
            flat_data = np.array(img_flatten)
        else:
            flat_data = np.append(flat_data, img_flatten, axis=0)

    data_w_coords = extract_coords(data)
    del data

    df = pd.DataFrame(flat_data)

    df["target"] = data_w_coords["target"]
    df["patient_id"] = data_w_coords["patient_id"]
    df["x"] = data_w_coords["x"]
    df["y"] = data_w_coords["y"]

    end = time.time()

    df.to_pickle(filename)
    print("Done Loading Images Dataframe. Took %ds" % (end - start))
    return


def flatten_image_and_new_df_small(data, filename):
    print("Beginning to Load Image Dataframe Subset")
    start = time.time()
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        img_path = row["path"]
        img_array = imread(img_path)
        img_resized = resize(img_array, (50, 50, 3))
        img_flatten = np.array(img_resized.flatten(), dtype=float).reshape(
            1, 7500
        )
        if i == 0:
            flat_data = np.array(img_flatten)
        else:
            flat_data = np.append(flat_data, img_flatten, axis=0)

    data_w_coords = extract_coords(data)
    del data

    df = pd.DataFrame(flat_data)

    df["target"] = data_w_coords["target"]
    df["patient_id"] = data_w_coords["patient_id"]
    df["x"] = data_w_coords["x"]
    df["y"] = data_w_coords["y"]

    end = time.time()

    df.to_pickle(filename)
    print(f"Done Loading Images Dataframe Subset.\n Took {end-start}s\n")
    return


# Reads inputs and determines which dataframes to load from dataset
def main(argv):
    load_small_set = False
    size = 10
    file_num = 1
    try:
        opts, _ = getopt.getopt(argv, "hsn:f:", [])
    except getopt.GetoptError:
        print(
            "svm_image_cancer_code.py -s\n\
                -s = load a smaller subset of the images\n\
                -n = number of patients to be loaded in"
        )
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "svm_image_cancer_code.py -s\n\
                -s = load a smaller subset of the images\n\
                -n = number of patients to be loaded in"
            )
            sys.exit()
        elif opt in ("-s", "--small"):
            load_small_set = True
        elif opt in ("-n", "--num_args"):
            size = int(arg)
        elif opt in ("-f", "--file_num"):
            file_num = int(arg)

    small_pickle_file = os.path.join(
        PICKLE_PATH,
        f"initial_dataframes/small_{size}_{file_num}_data_no_image.pkl",
    )
    pickle_file = os.path.join(
        PICKLE_PATH, "initial_dataframes/data_no_image.pkl"
    )

    small_img_pickle_file = os.path.join(
        PICKLE_PATH,
        f"initial_dataframes/small_{size}_{file_num}_img_pickle_file.pkl",
    )
    img_pickle_file = os.path.join(
        PICKLE_PATH, "initial_dataframes/img_pickle_file.pkl"
    )

    # Load and save either the small non image dataframe or whole thing
    if load_small_set:
        data = original_save_as_pickle_small(small_pickle_file, size)
    else:
        data = original_save_as_pickle(pickle_file)

    data["patient_id"] = data["patient_id"].astype(int)
    data["target"] = data["target"].astype(int)

    # Load and save either the small or full image dataset
    if load_small_set:
        flatten_image_and_new_df_small(data, small_img_pickle_file)
    else:
        flatten_image_and_new_df(data, img_pickle_file)


if __name__ == "__main__":
    main(sys.argv[1:])
