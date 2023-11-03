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
from collections import Counter  

sns.set()

from os import listdir
from skimage.transform import resize
from skimage.io import imread
from PIL import Image
from sklearn import svm
from sklearn.model_selection import GridSearchCV,cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix


FILE_DIR = os.listdir("/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/")
BASE_PATH = "/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/"
PICKLE_PATH = "/scratch/owebb1/"
FOLDER = os.listdir(BASE_PATH)


# DEFINITION: Main will load the images in if flagged
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

    if load_small_set:
        try:
            print("Attempting to read from %s" % (small_img_pickle_file))
            df = pd.read_pickle(small_img_pickle_file)
            print("Successfully Read Image Dataframe From Pickle File")
        except Exception:
            print(
                "ERROR: cannot read pickle file. Need to load in image dataframe file"
            )
            exit(2)
    else:
        try:
            print("Attempting to read from %s" % (img_pickle_file))
            df = pd.read_pickle(img_pickle_file)
            print("Successfully Read Image Dataframe From Pickle File")
        except Exception:
            print(
                "ERROR: cannot read pickle file. Need to load in image dataframe file"
            )
            exit(2)


    df = df.dropna()    

    # currently divides to make sure we have patients split
    df["patient_id"] = df["patient_id"].astype(int)

    df_patients = df.patient_id.unique()

    train_ids, sub_test_ids = train_test_split(df_patients, test_size=0.3, random_state=0)
    test_ids, dev_ids = train_test_split(sub_test_ids, test_size=0.5, random_state=0)

    
    # divy up the data into a train, test, and dev dataframes
    # TODO: Make this into a 50/50 split

    # Create the training data
    train_df_1 = df.loc[(data.patient_id.isin(train_ids)) & (data.target == 1), :]
    train_df_0_full = df.loc[(data.patient_id.isin(train_ids)) & (data.target == 0), :]
    length_train = Counter(train_df_1["target"])[1]
    train_df_0_eq = train_df_0_full.head(length_train)
    train_add_to_dev = train_df_0_full.iloc[length_train:]
    train_df = pd.concat([train_df_0_eq, train_df_1], ignore_index=True, sort=False)
    print(f"Train: {Counter(train_df['target'])}")

    # Create the testing data
    test_df_1 = df.loc[(data.patient_id.isin(test_ids)) & (data.target == 1), :]
    test_df_0_full = df.loc[(data.patient_id.isin(test_ids)) & (data.target == 0), :]
    length_test = Counter(test_df_1["target"])[1]
    test_df_0_eq = test_df_0_full.head(length_test)
    test_add_to_dev = test_df_0_full.iloc[length_test:]
    test_df = pd.concat([test_df_0_eq, test_df_1], ignore_index=True, sort=False)
    print(f"Test: {Counter(test_df['target'])}")

    # Create the development data - this should be a collection of garbage
    dev_df_norm = df.loc[data.patient_id.isin(dev_ids), :]
    dev_df = pd.concat([dev_df_norm, train_add_to_dev, test_add_to_dev])
    print(f"Dev: {Counter(dev_df['target'])}")

    # train_df = df.loc[data.patient_id.isin(train_ids)]
    # test_df = df.loc[data.patient_id.isin(test_ids)]
    # dev_df = df.loc[data.patient_id.isin(dev_ids)]


    del df

    # train_df_without_target = train_df[train_df.columns.difference(["target", "patient_id"])]
    c = 1
    kernal = "linear"
    svc = svm.SVC(class_weight='balanced',C=c, kernel=kernal)
    print(svc)

    param_grid=[{'C': [0.001,0.01,0.1], 'kernel': ['linear']},
        # {'C':[0.01],'gamma':[0.001],'kernel':['poly']},
        {'C': [0.001,0.01,0.1], 'gamma': [0.001], 'kernel': ['rbf']}
    ]


    # model=GridSearchCV(svc, param_grid, verbose = 3, n_jobs=-1)
   
    X_train = train_df[train_df.columns.difference(["target", "patient_id"])]
    y_train = train_df.target

    print("Starting SVC Training")
    start = time.time()
    svc.fit(X_train, y_train)
    # scores = cross_val_score(svc, X_train, y_train, cv=5)
    # print(scores)
    end = time.time()
    print("Finished SVC Training")
    print(f"Training took {end-start}s")

    
    X_test = test_df[test_df.columns.difference(["target", "patient_id"])]
    y_test = test_df.target
    
    # cm = confusion_matrix(y_test, predictions)
    # print(cm)
    # ax= plt.subplot()
    # sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')
    # ax.set_title('Confusion Matrix')
    # ax.xaxis.set_ticklabels(['business', 'health'])
    # ax.yaxis.set_ticklabels(['health', 'business'])


    predictions = svc.predict(X_test)
    # print(classification_report(y_test, predictions))

    # print("Best Params: " + str(model.best_params_))

    pred = pd.Series(predictions)

    print(pd.crosstab(y_test, pred).apply(lambda r: 100.0 * r/r.sum()))

    score = svc.score(X_test, y_test)
    rounded = round(score, 3)
    print("Accruacy Of The Test Set: ", rounded)

    with open(f'../scratch/{kernal}_{str(c)}_small.txt', 'w') as f:
        f.write(str(classification_report(y_test, predictions)))
        f.write
        #f.write("Best Params: " + str(model.best_params_))
        #f.write("Best Estimator: " + str(model.best_estimator_))

    filename = f'/scratch/owebb1/{kernal}_{str(c)}_{rounded}_small.pkl'
    pickle.dump(svc, open(filename, 'wb'))

if __name__ == "__main__":
    main(sys.argv[1:])