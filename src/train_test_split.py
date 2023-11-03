from lib2to3.pgen2.pgen import DFAState
import pickle
import warnings
import sys
import getopt
import pandas as pd
import os
import gc

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


FILE_DIR = os.listdir("/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/")
BASE_PATH = "/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/"
PICKLE_PATH = "/scratch/owebb1/"
FOLDER = os.listdir(BASE_PATH)
LOAD_SMALL_SET = False
SIZE = 10
FILE_NUM = 1


def set_args(args):
    global LOAD_SMALL_SET
    global SIZE
    global FILE_NUM
    try:
        opts, _ = getopt.getopt(args, "hsn:f:", [])
    except getopt.GetoptError:
        print(
            "svm_image_cancer_code.py -s\n\
                    -s = load a smaller subset of the images"
        )
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(
                "svm_image_cancer_code.py -s\n\
                    -s = load a smaller subset of the images"
            )
            sys.exit()
        elif opt == "-s":
            LOAD_SMALL_SET = True
        elif opt == "-n":
            SIZE = int(arg)
        elif opt == "-f":
            FILE_NUM = int(arg)


# DEFINITION: Main will load the images in if flagged
def main(argv):
    set_args(argv)
    print(LOAD_SMALL_SET)

    small_pickle_file = os.path.join(
        PICKLE_PATH,
        f"initial_dataframes/small_{SIZE}_{FILE_NUM}_data_no_image.pkl",
    )

    pickle_file = os.path.join(
        PICKLE_PATH, "initial_dataframes/data_no_image.pkl"
    )
    small_img_pickle_file = os.path.join(
        PICKLE_PATH,
        f"initial_dataframes/small_{SIZE}_{FILE_NUM}_img_pickle_file.pkl",
    )
    img_pickle_file = os.path.join(
        PICKLE_PATH, "initial_dataframes/img_pickle_file.pkl"
    )

    if LOAD_SMALL_SET:
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

    if LOAD_SMALL_SET:
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

    # define standard scaler
    # scaler = StandardScaler()
    # cols = [i for i in range(7500)]
    # # transform data
    # df[cols] = scaler.fit_transform(df[cols])

    df["patient_id"] = df["patient_id"].astype(int)

    df_patients = df.patient_id.unique()

    train_ids, sub_test_ids = train_test_split(
        df_patients, test_size=0.3, random_state=0
    )
    test_ids, dev_ids = train_test_split(
        sub_test_ids, test_size=0.5, random_state=0
    )

    # divy up the data into a train, test, and dev dataframes
    # 50/50 split

    # Create the training data
    train_df_1 = df.loc[
        (data.patient_id.isin(train_ids)) & (data.target == 1), :
    ]
    train_df_0_full = df.loc[
        (data.patient_id.isin(train_ids)) & (data.target == 0), :
    ]
    length_train = Counter(train_df_1["target"])[1]
    train_df_0_eq = train_df_0_full.head(length_train)
    train_add_to_dev = train_df_0_full.iloc[length_train:]
    train_df = pd.concat(
        [train_df_0_eq, train_df_1], ignore_index=True, sort=False
    )
    print(f"Train: {Counter(train_df['target'])}")

    # Create the testing data
    test_df_1 = df.loc[
        (data.patient_id.isin(test_ids)) & (data.target == 1), :
    ]
    test_df_0_full = df.loc[
        (data.patient_id.isin(test_ids)) & (data.target == 0), :
    ]
    length_test = Counter(test_df_1["target"])[1]
    test_df_0_eq = test_df_0_full.head(length_test)
    test_add_to_dev = test_df_0_full.iloc[length_test:]
    test_df = pd.concat(
        [test_df_0_eq, test_df_1], ignore_index=True, sort=False
    )
    print(f"Test: {Counter(test_df['target'])}")

    # Create the development data - this should be a collection of garbage
    dev_df_norm = df.loc[data.patient_id.isin(dev_ids), :]
    dev_df = pd.concat([dev_df_norm, train_add_to_dev, test_add_to_dev])
    print(f"Dev: {Counter(dev_df['target'])}")

    # Delete and collect garbage
    del df
    del train_df_1
    del train_df_0_full
    del length_train
    del train_df_0_eq
    del train_add_to_dev
    del dev_df_norm
    del test_df_1
    del test_df_0_full
    del test_df_0_eq
    del test_add_to_dev

    gc.collect()

    X_train = train_df[train_df.columns.difference(["target", "patient_id"])]
    y_train = train_df.target

    X_test = test_df[test_df.columns.difference(["target", "patient_id"])]
    y_test = test_df.target

    if LOAD_SMALL_SET:
        file_suffix = f"small_{SIZE}_{FILE_NUM}"
    else:
        file_suffix = "big"

    X_train_file = f"/scratch/owebb1/X_and_Y_test/X_train_{file_suffix}.pkl"
    pickle.dump(X_train, open(X_train_file, "wb"))
    y_train_file = f"/scratch/owebb1/X_and_Y_test/y_train_{file_suffix}.pkl"
    pickle.dump(y_train, open(y_train_file, "wb"))
    X_test_file = f"/scratch/owebb1/X_and_Y_test/X_test_{file_suffix}.pkl"
    pickle.dump(X_test, open(X_test_file, "wb"))
    y_test_file = f"/scratch/owebb1/X_and_Y_test/y_test_{file_suffix}.pkl"
    pickle.dump(y_test, open(y_test_file, "wb"))

    print(f"dumped into {X_train_file}")


if __name__ == "__main__":
    main(sys.argv[1:])
