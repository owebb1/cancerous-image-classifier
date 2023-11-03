import warnings
import sys
import getopt
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os import listdir
from skimage.io import imread
from sklearn.model_selection import train_test_split
from collections import Counter  

sns.set()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

FILE_DIR = os.listdir("/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/")
BASE_PATH = "/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/"
PICKLE_PATH = "/scratch/owebb1/"
FOLDER = os.listdir(BASE_PATH)


# DEFINITION: This will extract the coordinates from the filepath names
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
    path = BASE_PATH + str(patient_id) + "/" + str(cancer_id)
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


# DEFINITION: simply combines all of one patient to print image and overlay 
# of where cancer is
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



# DEFINITION: Plot the histogram fequencies for patients
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


# DEFINITION: Plot the positive labeled 50x50 images
def plot_pos_selection(data):
    pos_selection = np.random.choice(
        data[data.target == 1].index.values, size=50, replace=False
    )
    fig, ax = plt.subplots(5, 10, figsize=(20, 10))
    fig.suptitle('cancerous examples', fontsize=16)

    for n in range(5):
        for m in range(10):
            idx = pos_selection[m + 10 * n]
            image = imread(data.loc[idx, "path"])
            ax[n, m].imshow(image)
            ax[n, m].grid(False)

    plt.show()
    return


# DEFINITION: Plot the negative labeled 50x50 images
def plot_neg_selection(data):
    neg_selection = np.random.choice(
        data[data.target == 0].index.values, size=50, replace=False
    )
    fig, ax = plt.subplots(5, 10, figsize=(20, 10))
    fig.suptitle('Benign Examples', fontsize=16)

    for n in range(5):
        for m in range(10):
            idx = neg_selection[m + 10 * n]
            image = imread(data.loc[idx, "path"])
            ax[n, m].imshow(image)
            ax[n, m].grid(False)

    plt.show()
    return


# DEFINITION: Plot the x and y values of the patients
def plot_x_y_values(data):
    # example = get_patient_dataframe(str(data.patient_id.values[0]))
    fig, ax = plt.subplots(5, 3, figsize=(15, 22),constrained_layout=True)
    fig.suptitle('x and y values', fontsize=16)

    patient_ids = data.patient_id.unique()

    # will print out 15 patient values to establish x and y importance
    for n in range(5):
        for m in range(3):
            patient_id = patient_ids[m + 3 * n]
            example_df = get_patient_dataframe(str(patient_id))

            ax[n, m].scatter(
                example_df.x.values,
                example_df.y.values,
                c=example_df.target.values,
                cmap="coolwarm",
                s=20,
            )
            ax[n, m].set_title("patient " + str(patient_id))
            ax[n, m].set_xlabel("y coord")
            ax[n, m].set_ylabel("x coord")
    plt.show()
    return


# DEFINITION: Visualize one persons example
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
    plt.show()
    return



# DEFINITION: plot the amounts of train, dev, test splits
def plot_train_test_dev(train_df, dev_df, test_df):
    _, ax = plt.subplots(1, 3, figsize=(20, 5))
    sns.countplot(train_df.target, ax=ax[0], palette="Reds")
    ax[0].set_title("Train data")
    sns.countplot(test_df.target, ax=ax[1], palette="Greens")
    ax[1].set_title("Test data")
    sns.countplot(dev_df.target, ax=ax[2], palette="Blues")
    ax[2].set_title("Dev data")
    plt.show()


# DEFINITION: Creates visuals for each of the pieces
def main(argv):
    load_small_set = False
    try:
        opts, _ = getopt.getopt(argv, "hs", [])
    except getopt.GetoptError:
        print(
            "svm_image_cancer_code.py -s\n\
                    -s = load a smaller subset of the images"
        )
        sys.exit(2)
    for opt, _ in opts:
        if opt == "-h":
            print(
                "svm_image_cancer_code.py -s\n\
                    -s = load a smaller subset of the images"
            )
            sys.exit()
        elif opt in ("-s", "--small"):
            load_small_set = True


    small_pickle_file = os.path.join(PICKLE_PATH, "small_data_no_image.pkl")
    pickle_file = os.path.join(PICKLE_PATH, "data_no_image.pkl")
    small_img_pickle_file = os.path.join(PICKLE_PATH, "small_img_pickle_file.pkl")
    img_pickle_file = os.path.join(PICKLE_PATH, "img_pickle_file.pkl")

    if load_small_set:
        try:
            print(f"Attempting to read from {small_pickle_file}")
            data = pd.read_pickle(small_pickle_file)
            print("Successfully Read Non Image Dataframe From Pickle File")
        except Exception:
            print(
                "ERROR: cannot read pickle file. \
                Need to load in original dataframe file"
            )
            exit(2)
    else:
        try:
            print(f"Attempting to read from {pickle_file}")
            data = pd.read_pickle(pickle_file)
            print("Successfully Read Non Image Dataframe From Pickle File")
        except Exception:
            print(
                "ERROR: cannot read pickle file. \
                Need to load in original dataframe file"
            )
            exit(2)

    # convert to make data more compressed
    data["patient_id"] = data["patient_id"].astype(int)
    data["target"] = data["target"].astype(int)

    # if load_small_set:
    #     try:
    #         print("Attempting to read from %s" % (small_img_pickle_file))
    #         df = pd.read_pickle(small_img_pickle_file)
    #         print("Successfully Read Image Dataframe From Pickle File")
    #     except Exception:
    #         print(
    #             "ERROR: cannot read pickle file. Need to load in image dataframe file"
    #         )
    #         exit(2)
    # else:
    #     try:
    #         print("Attempting to read from %s" % (img_pickle_file))
    #         df = pd.read_pickle(img_pickle_file)
    #         print("Successfully Read Image Dataframe From Pickle File")
    #     except Exception:
    #         print(
    #             "ERROR: cannot read pickle file. Need to load in image dataframe file"
    #         )
    #         exit(2)


    # df = df.dropna()    
    # df["patient_id"] = df["patient_id"].astype(int)
    
    df_patients = data.patient_id.unique()


    train_ids, sub_test_ids = train_test_split(df_patients, test_size=0.3, random_state=0)
    test_ids, dev_ids = train_test_split(sub_test_ids, test_size=0.5, random_state=0)
    

    train_df = data.loc[data.patient_id.isin(train_ids), :]
    test_df = data.loc[data.patient_id.isin(test_ids), :]
    dev_df = data.loc[data.patient_id.isin(dev_ids), :]



    train_df_1 = data.loc[(data.patient_id.isin(train_ids)) & (data.target == 1), :]
    train_df_0_full = data.loc[(data.patient_id.isin(train_ids)) & (data.target == 0), :]
    length_train = Counter(train_df_1["target"])[1]
    train_df_0_eq = train_df_0_full.head(length_train)
    train_add_to_dev = train_df_0_full.iloc[length_train:]
    train_df = pd.concat([train_df_0_eq, train_df_1], ignore_index=True, sort=False)
    print(f"Train: {Counter(train_df['target'])}")

    # Create the testing data
    test_df_1 = data.loc[(data.patient_id.isin(test_ids)) & (data.target == 1), :]
    test_df_0_full = data.loc[(data.patient_id.isin(test_ids)) & (data.target == 0), :]
    length_test = Counter(test_df_1["target"])[1]
    test_df_0_eq = test_df_0_full.head(length_test)
    test_add_to_dev = test_df_0_full.iloc[length_test:]
    test_df = pd.concat([test_df_0_eq, test_df_1], ignore_index=True, sort=False)
    print(f"Test: {Counter(test_df['target'])}")

    # Create the development data - this should be a collection of garbage
    dev_df_norm = data.loc[data.patient_id.isin(dev_ids), :]
    dev_df = pd.concat([dev_df_norm, train_add_to_dev, test_add_to_dev])
    print(f"Dev: {Counter(dev_df['target'])}")

    print("Plots Are On")

    # plot_hist_frequencies(data)
    # plot_pos_selection(data)
    # plot_neg_selection(data)
    # plot_x_y_values(data)
    # visualize_one_example()
    plot_train_test_dev(train_df, dev_df, test_df)

if __name__ == "__main__":
    main(sys.argv[1:])




