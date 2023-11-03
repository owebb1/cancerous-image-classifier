import pickle

# from msilib.schema import Error
from pickle import PickleError
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import sys
import getopt
import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set()

from os import listdir
from skimage.transform import resize
from skimage.io import imread
from PIL import Image
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix




FILE_DIR = os.listdir("/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/")
BASE_PATH = "/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/"
PICKLE_PATH = "/scratch/owebb1/"
FOLDER = os.listdir(BASE_PATH)

"""
DEFINITION: This will extract the coordinates from the filepath names
"""
# Will extract x and y coordinates from the path to file
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


"""
DEFINITION: Creates a new dataframe and puts image data in it
    Columns = target, path, x, y
"""


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


"""
DEFINITION: Simply calls for certain patient for positive and negative 
"""


def get_patient_dataframe(patient_id):
    df_0 = get_cancer_dataframe(patient_id, "0")
    df_1 = get_cancer_dataframe(patient_id, "1")
    patient_df = df_0.append(df_1)
    return patient_df


"""
DEFINITION: simply combines all of one patient to print image and overlay 
of where cancer is
"""


def visualise_breast_tissue(patient_id, pred_df=None):
    example_df = get_patient_dataframe(patient_id)
    max_point = [example_df.y.max() - 1, example_df.x.max() - 1]
    grid = 255 * np.ones(shape=(max_point[0] + 50, max_point[1] + 50, 3)).astype(
        np.uint8
    )
    mask = 255 * np.ones(shape=(max_point[0] + 50, max_point[1] + 50, 3)).astype(
        np.uint8
    )
    if pred_df is not None:
        patient_df = pred_df[pred_df.patient_id == patient_id].copy()
    mask_proba = np.zeros(shape=(max_point[0] + 50, max_point[1] + 50, 1)).astype(
        np.float
    )

    broken_patches = []
    for n in range(len(example_df)):
        try:
            image = imread(example_df.path.values[n])

            target = example_df.target.values[n]

            x_coord = np.int(example_df.x.values[n])
            y_coord = np.int(example_df.y.values[n])
            x_start = x_coord - 1
            y_start = y_coord - 1
            x_end = x_start + 50
            y_end = y_start + 50

            grid[y_start:y_end, x_start:x_end] = image
            if target == 1:
                mask[y_start:y_end, x_start:x_end, 0] = 250
                mask[y_start:y_end, x_start:x_end, 1] = 0
                mask[y_start:y_end, x_start:x_end, 2] = 0
            if pred_df is not None:

                proba = patient_df[
                    (patient_df.x == x_coord) & (patient_df.y == y_coord)
                ].proba
                mask_proba[y_start:y_end, x_start:x_end, 0] = np.float(proba)

        except ValueError:
            broken_patches.append(example_df.path.values[n])

    return grid, mask, broken_patches, mask_proba


"""
DEFINITION: Load the image data into the dataframes and save it as a pkl file
"""


def load_dataframes_from_raw_save_as_pickle(filename):
    target_arr = []
    # datadir='/Users/owenwebb/E90_data/archive-2/IDC_regular_ps50_idx5'
    patient_id_arr = []
    path_arr = []

    for n in range(len(FOLDER)):
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

    return data

def load_dataframes_from_raw_save_as_pickle_small(small_img_pkl, size):
    target_arr = []
    # datadir='/Users/owenwebb/E90_data/archive-2/IDC_regular_ps50_idx5'
    patient_id_arr = []
    path_arr = []

    for n in range(size): #len(FOLDER)):
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
    patient_id_np = np.array(patient_id_arr, dtype=int)
    path_arr_np = np.array(path_arr)
    data = pd.DataFrame()
    data["target"] = target
    data["patient_id"] = patient_id_np
    data["path"] = path_arr_np

    data.to_pickle(small_img_pkl)

    return patient_id_arr, data


"""
DEFINITION: Counts the total images in the directory
"""


def print_num_images():
    total_images = 0
    for n in range(len(FOLDER)):
        patient_id = FOLDER[n]
        for c in [0, 1]:
            patient_path = BASE_PATH + patient_id
            class_path = patient_path + "/" + str(c) + "/"
            subfiles = os.listdir(class_path)
            total_images += len(subfiles)

    print("total number of images = %d" % total_images)
    return


"""
DEFINITION: Plot the histogram fequencies for patients
"""


def plot_hist_frequencies(data):
    cancer_perc = (
        data.groupby("patient_id").target.value_counts()
        / data.groupby("patient_id").target.size()
    )
    cancer_perc = cancer_perc.unstack()
    _, ax = plt.subplots(1, 3, figsize=(20, 5))
    sns.distplot(
        data.groupby("patient_id").size(), ax=ax[0], color="Orange", kde=False, bins=30
    )
    ax[0].set_xlabel("Number of patches")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("How many patches do we have per patient?")
    sns.distplot(
        cancer_perc.loc[:, 1] * 100, ax=ax[1], color="Tomato", kde=False, bins=30
    )
    ax[1].set_title("How much percentage of an image is covered by IDC?")
    ax[1].set_ylabel("Frequency")
    ax[1].set_xlabel("'%' of patches with IDC")
    sns.countplot(data.target, palette="Set2", ax=ax[2])
    ax[2].set_xlabel("no(0) versus yes(1)")
    ax[2].set_title("How many patches show IDC?")
    return


"""
DEFINITION: Plot the positive labeled 50x50 images
"""


def plot_pos_selection(data):
    pos_selection = np.random.choice(
        data[data.target == 1].index.values, size=50, replace=False
    )
    fig, ax = plt.subplots(5, 10, figsize=(20, 10))

    for n in range(5):
        for m in range(10):
            idx = pos_selection[m + 10 * n]
            image = imread(data.loc[idx, "path"])
            ax[n, m].imshow(image)
            ax[n, m].grid(False)
    return


"""
DEFINITION: Plot the negative labeled 50x50 images
"""


def plot_neg_selection(data):
    neg_selection = np.random.choice(
        data[data.target == 0].index.values, size=50, replace=False
    )
    fig, ax = plt.subplots(5, 10, figsize=(20, 10))

    for n in range(5):
        for m in range(10):
            idx = neg_selection[m + 10 * n]
            image = imread(data.loc[idx, "path"])
            ax[n, m].imshow(image)
            ax[n, m].grid(False)
    return


"""
DEFINITION: Plot the x and y values of the patients
"""


def plot_x_y_values(data):
    example = get_patient_dataframe(data.patient_id.values[0])

    _, ax = plt.subplots(5, 3, figsize=(20, 27))

    patient_ids = data.patient_id.unique()

    # will print out 15 patient values to establish x and y importance
    for n in range(5):
        for m in range(3):
            patient_id = patient_ids[m + 3 * n]
            example_df = get_patient_dataframe(patient_id)

            ax[n, m].scatter(
                example_df.x.values,
                example_df.y.values,
                c=example_df.target.values,
                cmap="coolwarm",
                s=20,
            )
            ax[n, m].set_title("patient " + patient_id)
            ax[n, m].set_xlabel("y coord")
            ax[n, m].set_ylabel("x coord")
    return


"""
DEFINITION: Visualize one persons example
"""


def visualize_one_example():
    example = "13616"
    grid, mask, broken_patches, _ = visualise_breast_tissue(example)

    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(grid, alpha=0.9)
    ax[1].imshow(mask, alpha=0.8)
    ax[1].imshow(grid, alpha=0.7)
    ax[0].grid(False)
    ax[1].grid(False)
    for m in range(2):
        ax[m].set_xlabel("y-coord")
        ax[m].set_ylabel("y-coord")
    ax[0].set_title("Breast tissue slice of patient: " + example)
    ax[1].set_title("Cancer tissue colored red \n of patient: " + example)
    return


"""
DEFINITION: flatten the image and put it into a large dataframe
"""


def flatten_image_and_new_df(data, filename):
    start = time.time()

    # flat_data = np.array([],dtype=float)
    i = 1
    for img_path in data.path:
        img_array = imread(img_path)
        img_resized = resize(img_array, (50, 50, 3))
        img_flatten = np.array(img_resized.flatten(), dtype=float)
        if i == 1:
            flat_data = np.array(img_flatten)
        else:
            flat_data = np.append(flat_data, img_flatten, 0)
        i += 1

    end = time.time()
    print("create flat data = %f" % (end - start))

    data_w_coords = extract_coords(data)
    del data

    print("Done with data with coords")

    # creates a dataframe that contains the images as rows with
    # features at columns

    # This is where the memory jumps way too high
    # Need to load in data and delete

    df = pd.DataFrame(flat_data)
    print(df.head())
    df["target"] = data_w_coords["target"]
    df["patient_id"] = data_w_coords["patient_id"]
    df["x"] = data_w_coords["x"]
    df["y"] = data_w_coords["y"]

    df.to_pickle(filename)
    return df


"""
DEFINITION: plot the amounts of train, dev, test splits
"""
def plot_train_test_dev(train_df, dev_df, test_df):
    _, ax = plt.subplots(1, 3, figsize=(20, 5))
    sns.countplot(train_df.target, ax=ax[0], palette="Reds")
    ax[0].set_title("Train data")
    sns.countplot(dev_df.target, ax=ax[1], palette="Blues")
    ax[1].set_title("Dev data")
    sns.countplot(test_df.target, ax=ax[2], palette="Greens")
    ax[2].set_title("Test data")


def flatten_image_and_new_df_small(data, filename):
    start = time.time()

    # num_rows = len(data.patient_id)
        
    
    for i, row in tqdm(data.iterrows(), total=data.shape[0]): 
        img_path = row["path"]
        img_array = imread(img_path)
        img_resized = resize(img_array, (50, 50, 3))
        img_flatten = np.array(img_resized.flatten(), dtype=float).reshape(1,7500)
        if i == 0:
            flat_data = np.array(img_flatten)
            # print(flat_data)
            # flat_data.reshape((7500, 1))
        else:
            flat_data = np.append(flat_data, img_flatten,axis=0)

    end = time.time()
    print("create flat data = %f" % (end - start))

    data_w_coords = extract_coords(data)
    del data

    print("Done with data with coords")

    # creates a dataframe that contains the images as rows with
    # features at columns

    # This is where the memory jumps way too high
    # Need to load in data and delete

    df = pd.DataFrame(flat_data)
    # print("made it here!")
    df["target"] = data_w_coords["target"]
    df["patient_id"] = data_w_coords["patient_id"]
    df["x"] = data_w_coords["x"]
    df["y"] = data_w_coords["y"]

    print(df.head())
    print(len(df))
    print(len(data_w_coords))

    df.to_pickle(filename)
    return df


"""
DEFINITION: Main will load the images in if flagged
"""
def main(argv):
    load_imgs = False
    load_original_df = False
    plots = False
    load_small_set = False
    try:
        opts, args = getopt.getopt(argv, "hiops", [])
    except getopt.GetoptError:
        print(
            "svm_image_cancer_code.py -i -o\n\
                    -i = load all the images again\n\
                    -o = load the original dataframe\n\
                    -p = plot the images\n\
                    -s = load a smaller subset of the images"
        )
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "svm_image_cancer_code.py -i -o\n\
                    -i = load all the images again\n\
                    -o = load the original dataframe\n\
                    -p = plot the images\n\
                    -s = load a smaller subset of the images"
            )
            sys.exit()
        elif opt in ("-i", "--images"):
            load_imgs = True
        elif opt in ("-o", "--originaldf"):
            load_original_df = True
        elif opt in ("-p", "--plots"):
            plots = True
        elif opt in ("-s", "--small"):
            load_small_set = True

    print_num_images()
    small_pickle_file = os.path.join(PICKLE_PATH, "small_data_no_image.pkl")
    pickle_file = os.path.join(PICKLE_PATH, "data_no_image.pkl")
    # load_data.py file

    if load_small_set:
        patient_id_arr, data = load_dataframes_from_raw_save_as_pickle_small(small_pickle_file, 10)
    elif load_original_df:
        print("Begining to Load Original Dataframe")
        data = load_dataframes_from_raw_save_as_pickle(pickle_file)
        print("Done Loading Original Dataframe")
    elif not load_small_set:
        try:
            print("Attempting to read from %s" % (pickle_file))
            data = pd.read_pickle(pickle_file)
            print("Successfully Read Original Dataframe From Pickle File")
        except Exception:
            print(
                "ERROR: cannot read pickle file. \
            Need to load in original dataframe file"
            )
            exit(2)

    if plots:
        print("Plots Are On")
        plot_hist_frequencies(data)
        plot_pos_selection(data)
        plot_neg_selection(data)
        plot_x_y_values(data)
        visualize_one_example()

    # convert to make data more compressed
    data["patient_id"] = data["patient_id"].astype(int)
    data["target"] = data["target"].astype(int)
    small_img_pickle_file = os.path.join(PICKLE_PATH, "small_img_pickle_file.pkl")
    img_pickle_file = os.path.join(PICKLE_PATH, "img_pickle_file.pkl")

    small_off = False
    if load_small_set:
        df = flatten_image_and_new_df_small(data, small_img_pickle_file)
    elif small_off:
        try:
            print("Attempting to read from %s" % (small_img_pickle_file))
            df = pd.read_pickle(small_img_pickle_file)
            print("Successfully Read Image Dataframe From Pickle File")
        except Exception:
            print(
                "ERROR: cannot read pickle file. Need to load in image dataframe file"
            )
            exit(2)

    if load_imgs:
        print("Beginning to Load Images Dataframe")
        start = time.time()
        df = flatten_image_and_new_df(data, img_pickle_file)
        end = time.time()
        print("Done Loading Images Dataframe. Took %ds" % (end - start))
    elif not small_off:
        try:
            print("Attempting to read from %s" % (img_pickle_file))
            df = pd.read_pickle(img_pickle_file)
            print("Successfully Read Image Dataframe From Pickle File")
        except Exception:
            print(
                "ERROR: cannot read pickle file. Need to load in image dataframe file"
            )
            exit(2)


    # df.drop(columns=0, inplace=True)

    # split based on patient_id, so that we can train on all the x and y's
    # patients = data.patient_id.unique()
    df = df.dropna()    
    df["patient_id"] = df["patient_id"].astype(int)
    
    df_patients = df.patient_id.unique()

    # print("DF Patients")
    # print("*"*10)
    # for patient in df_patients:
    #     print(patient)
    
    # print("Data Patients")
    # print("*"*10)
    # for patient in patients:
    #     print(patient)

    train_ids, sub_test_ids = train_test_split(df_patients, test_size=0.3, random_state=0)
    test_ids, dev_ids = train_test_split(sub_test_ids, test_size=0.5, random_state=0)

    # divy up the data into a train, test, and dev dataframes
    start = time.time()
    train_df = df.loc[data.patient_id.isin(train_ids), :]
    test_df = df.loc[data.patient_id.isin(test_ids), :]
    dev_df = df.loc[data.patient_id.isin(dev_ids), :]
    end = time.time()

    print(train_df.head())
    print(train_df.shape)

    print("time: %f" % (end - start))
    if plots:
        plot_train_test_dev(train_df, dev_df, test_df)

    train_df_without_target = train_df[train_df.columns.difference(["target", "patient_id"])]
    
    svc = svm.SVC()

    param_grid={
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        # {'C':[0.01],'gamma':[0.001],'kernel':['poly']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    }

    model=GridSearchCV(svc, param_grid, verbose = 3, n_jobs=-1)
   
    train_df_without_target = train_df[train_df.columns.difference(["target", "patient_id"])]

    print("Starting SVC Training")
    svc.fit(train_df_without_target, train_df.target)

    print("Finished SVC Training")

    X_test = test_df[test_df.columns.difference(["target", "patient_id"])]
    y_test = test_df.target

    predictions = svc.predict(X_test)

    with open('../scratch/run3.txt', 'w') as f:
        f.write(str(classification_report(y_test, predictions)))
        #f.write("Best Params: " + str(model.best_params_))
        #f.write("Best Estimator: " + str(model.best_estimator_))

    # score = model.score(test_df[test_df.columns.difference(["target", "patient_id"])], test_df.target)
    
    
    # print("Accruacy Of The Test Set: ", score)
    filename = '/scratch/owebb1/smv_3.sav'
    pickle.dump(svc, open(filename, 'wb'))
    


if __name__ == "__main__":
    main(sys.argv[1:])
