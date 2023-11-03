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
sns.set()

from os import listdir
from skimage.transform import resize
from skimage.io import imread
from PIL import Image
from matplotlib.pyplot import imread
from skimage.io import imread
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


FILE_DIR = os.listdir("/Users/owenwebb/E90_data/archive-2/IDC_regular_ps50_idx5/")
BASE_PATH = "/Users/owenwebb/E90_data/archive-2/IDC_regular_ps50_idx5/"
PICKLE_PATH = "/Users/owenwebb/E90_data/"
FOLDER = os.listdir(BASE_PATH)

"""
DEFINITION: 
"""
# Will extract x and y coordinates from the path to file
def extract_coords(df):
    coord = df.path.str.rsplit("_", n=4, expand=True)
    coord = coord.drop([0, 1, 4], axis=1)
    coord = coord.rename({2: "x", 3: "y"}, axis=1)
    coord.loc[:, "x"] = coord.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)
    coord.loc[:, "y"] = coord.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)
    df.loc[:, "x"] = coord.x.values
    df.loc[:, "y"] = coord.y.values
    return df

"""
DEFINITION: 
"""
# creates a new dataframe for each patient
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
    dataframe.loc[:, "x"] = dataframe.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)
    dataframe.loc[:, "y"] = dataframe.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)
    return dataframe

"""
DEFINITION: 
"""
# Simply calls for certain patient for positive and negative
def get_patient_dataframe(patient_id):
    df_0 = get_cancer_dataframe(patient_id, "0")
    df_1 = get_cancer_dataframe(patient_id, "1")
    patient_df = df_0.append(df_1)
    return patient_df

"""
DEFINITION: 
"""
# simply combines all of one patient to print image and overlay of where cancer is
def visualise_breast_tissue(patient_id, pred_df=None):
    example_df = get_patient_dataframe(patient_id)
    max_point = [example_df.y.max()-1, example_df.x.max()-1]
    grid = 255*np.ones(shape = (max_point[0] + 50, max_point[1] + 50, 3)).astype(np.uint8)
    mask = 255*np.ones(shape = (max_point[0] + 50, max_point[1] + 50, 3)).astype(np.uint8)
    if pred_df is not None:
        patient_df = pred_df[pred_df.patient_id == patient_id].copy()
    mask_proba = np.zeros(shape = (max_point[0] + 50, max_point[1] + 50, 1)).astype(np.float)
    
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
                    (patient_df.x==x_coord) & (patient_df.y==y_coord)].proba
                mask_proba[y_start:y_end, x_start:x_end, 0] = np.float(proba)

        except ValueError:
            broken_patches.append(example_df.path.values[n])
    
    
    return grid, mask, broken_patches, mask_proba

"""
DEFINITION: 
"""
# Load the image data into the dataframes
def load_dataframes_from_raw_save_as_pickle(filename):
    target_arr=[]
    # datadir='/Users/owenwebb/E90_data/archive-2/IDC_regular_ps50_idx5'
    patient_id_arr = []
    path_arr = []

    for n in range(len(FOLDER)):
        patient_id = FOLDER[n]
        patient_path = BASE_PATH + patient_id 
        for c in [0,1]:
            class_path = patient_path + "/" + str(c) + "/"
            subfiles = os.listdir(class_path)
            for img in range(len(subfiles)):
                image_path = subfiles[img]
                complete_path = os.path.join(class_path,image_path)
                target_arr.append(c)
                patient_id_arr.append(patient_id)
                path_arr.append(complete_path)
        
    target=np.array(target_arr)
    patient_id_np = np.array(patient_id_arr)
    path_arr_np = np.array(path_arr)
    data=pd.DataFrame()
    data["target"]=target
    data["patient_id"] = patient_id_np
    data["path"] = path_arr_np

    data.to_pickle(filename)

    return data

"""
DEFINITION: 
"""
# Counts the total images in the files
def print_num_images():
    total_images = 0
    for n in range(len(FOLDER)):
        patient_id = FOLDER[n]
        for c in [0, 1]:
            patient_path = BASE_PATH + patient_id 
            class_path = patient_path + "/" + str(c) + "/"
            subfiles = os.listdir(class_path)
            total_images += len(subfiles)
            
    print("total number of images = %d" %total_images)
    return

"""
DEFINITION: 
"""
# Plot the histogram fequencies for patients
def plot_hist_frequencies(data):
    cancer_perc = data.groupby("patient_id").target.value_counts()/ data.groupby("patient_id").target.size()
    cancer_perc = cancer_perc.unstack()
    _, ax = plt.subplots(1,3,figsize=(20,5))
    sns.distplot(data.groupby("patient_id").size(), ax=ax[0], color="Orange", kde=False, bins=30)
    ax[0].set_xlabel("Number of patches")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("How many patches do we have per patient?")
    sns.distplot(cancer_perc.loc[:, 1]*100, ax=ax[1], color="Tomato", kde=False, bins=30)
    ax[1].set_title("How much percentage of an image is covered by IDC?")
    ax[1].set_ylabel("Frequency")
    ax[1].set_xlabel("'%' of patches with IDC")
    sns.countplot(data.target, palette="Set2", ax=ax[2])
    ax[2].set_xlabel("no(0) versus yes(1)")
    ax[2].set_title("How many patches show IDC?")
    return 

"""
DEFINITION: 
"""
def plot_pos_selection(data):
    pos_selection = np.random.choice(data[data.target==1].index.values, size=50, replace=False)
    fig, ax = plt.subplots(5,10,figsize=(20,10))

    for n in range(5):
        for m in range(10):
            idx = pos_selection[m + 10*n]
            image = imread(data.loc[idx, "path"])
            ax[n,m].imshow(image)
            ax[n,m].grid(False)
    return

"""
DEFINITION: 
"""
def plot_neg_selection(data):
    neg_selection = np.random.choice(data[data.target==0].index.values, size=50, replace=False)
    fig, ax = plt.subplots(5,10,figsize=(20,10))

    for n in range(5):
        for m in range(10):
            idx = neg_selection[m + 10*n]
            image = imread(data.loc[idx, "path"])
            ax[n,m].imshow(image)
            ax[n,m].grid(False)
    return

"""
DEFINITION: 
"""
def plot_x_y_values(data):
    # Will get an examle dataframe for the first patient
    example = get_patient_dataframe(data.patient_id.values[0])
    # example.head()

    _, ax = plt.subplots(5,3,figsize=(20, 27))

    patient_ids = data.patient_id.unique()

    # will print out 15 patient values to establish x and y importance
    for n in range(5):
        for m in range(3):
            patient_id = patient_ids[m + 3*n]
            example_df = get_patient_dataframe(patient_id)
            
            ax[n,m].scatter(example_df.x.values, example_df.y.values, c=example_df.target.values, cmap="coolwarm", s=20);
            ax[n,m].set_title("patient " + patient_id)
            ax[n,m].set_xlabel("y coord")
            ax[n,m].set_ylabel("x coord")
    return

"""
DEFINITION: 
"""
def visualize_one_example():
    example = "13616"
    grid, mask, broken_patches,_ = visualise_breast_tissue(example)

    _, ax = plt.subplots(1,2,figsize=(20,10))
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
DEFINITION: 
"""
def flatten_image_and_new_df(data, filename):
    start = time.time()
    flat_data_arr = []

    for img_path in data.path:
        img_array=imread(img_path)
        img_resized=resize(img_array,(50,50,3))
        flat_data_arr.append(img_resized.flatten())

    flat_data=np.array(flat_data_arr, dtype=np.float)

    end = time.time()
    print("create flat data = %f" %(end-start))

    data = extract_coords(data)
    
    # creates a dataframe that contains the images as rows with features at columns
    df = pd.DataFrame(flat_data)
    df["target"] = data["target"]
    df["patient_id"] = data["patient_id"]
    df["x"] = data["x"]
    df["y"] = data["y"]

    df.to_pickle(filename)
    return df

def plot_train_test_dev(train_df, dev_df, test_df):
    _, ax = plt.subplots(1,3,figsize=(20,5))
    sns.countplot(train_df.target, ax=ax[0], palette="Reds")
    ax[0].set_title("Train data")
    sns.countplot(dev_df.target, ax=ax[1], palette="Blues")
    ax[1].set_title("Dev data")
    sns.countplot(test_df.target, ax=ax[2], palette="Greens")
    ax[2].set_title("Test data")


"""
DEFINITION: 
"""
def main(argv):
    load_imgs = False
    load_original_df = False
    plots = False
    try:
        opts, args = getopt.getopt(argv,"hiop",[])
    except getopt.GetoptError:
        print ("svm_image_cancer_code.py -i -o\n\
                    -i = load all the images again\n\
                    -o = load the original dataframe\n\
                    -p = plot the images")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ("svm_image_cancer_code.py -i -o\n\
                    -i = load all the images again\n\
                    -o = load the original dataframe\n\
                    -p = plot the images")
            sys.exit()
        elif opt in ("-i", "--images"):
            load_imgs = True
        elif opt in ("-o", "--originaldf"):
            load_original_df = True
        elif opt in ("-p", "--plots"):
            plots = True

    print_num_images()

    pickle_file = os.path.join(PICKLE_PATH, "data_no_image.pkl")
    # load_data.py file
    
    if load_original_df:
        print("Begining to Load Original Dataframe")
        data = load_dataframes_from_raw_save_as_pickle(pickle_file)
        print("Done Loading Original Dataframe")
    else:
        try:
            print("Attempting to read from %s" %(pickle_file))
            data = pd.read_pickle(pickle_file)
            print("Successfully Read Original Dataframe From Pickle File")
        except Exception:
            print("ERROR: cannot read pickle file. Need to load in original dataframe file")
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


    img_pickle_file = os.path.join(PICKLE_PATH, "img_pickle_file.pkl")
    if load_imgs:
        print("Begining to Load Images Dataframe")
        start = time.time()
        df = flatten_image_and_new_df(data, img_pickle_file)
        end = time.time()
        print("Done Loading Images Dataframe. Took %ds" %(end-start))
    else:
        try:
            print("Attempting to read from %s" %(img_pickle_file))
            df = pd.read_pickle(img_pickle_file)
            print("Successfully Read Image Dataframe From Pickle File")
        except Exception:
            print("ERROR: cannot read pickle file. Need to load in image dataframe file")
            exit(2)


    # split based on patient_id, so that we can train on all the x and y's
    patients = data.patient_id.unique()

    train_ids, sub_test_ids = train_test_split(patients,
                                            test_size=0.3,
                                            random_state=0)
    test_ids, dev_ids = train_test_split(sub_test_ids, test_size=0.5, random_state=0)

    # divy up the data into a train, test, and dev dataframes
    start = time.time()
    train_df = df.loc[data.patient_id.isin(train_ids),:]
    test_df = df.loc[data.patient_id.isin(test_ids),:]
    dev_df = df.loc[data.patient_id.isin(dev_ids),:]
    end = time.time()
    
    print("time: %f" %(end-start))
    if plots:
        plot_train_test_dev(train_df, dev_df, test_df)

    # param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}

    svc=svm.SVC(probability=True)

    # model=GridSearchCV(svc,param_grid,scoring='accuracy')

    print("Starting SVC Training")
    svc.fit(train_df[train_df.columns.difference(["target","patient_id"])], train_df.target)
    print("Finished SVC Training")

    # histogram of the features
    # histogram of the positive and negative
    # PCA could reduce 
    # luminance and chromanance
    # 8-bit pixel depth vs np.float



if __name__ == "__main__":
    main(sys.argv[1:])