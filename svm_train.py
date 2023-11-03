import pickle
import warnings
import sys
import getopt
import pandas as pd
import os
import time

from sklearn import svm
from sklearn.ensemble import BaggingClassifier


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
        elif opt in ("-s", "--small"):
            LOAD_SMALL_SET = True
        elif opt in ("-n", "--num_args"):
            SIZE = int(arg)
        elif opt in ("-f"):
            FILE_NUM = int(arg)


# DEFINITION: Main will load the images in if flagged
def main(argv):
    set_args(argv)

    # determine whether to use big or small dataset
    if LOAD_SMALL_SET == True:
        file_suffix = f"small_{SIZE}_{FILE_NUM}"
    else:
        file_suffix = "big"

    # open the training data
    X_train_file = f"/scratch/owebb1/X_and_Y_test/X_train_{file_suffix}.pkl"
    X_train = pickle.load(open(X_train_file, "rb"))
    # print(f"loaded {X_train_file}")
    y_train_file = f"/scratch/owebb1/X_and_Y_test/y_train_{file_suffix}.pkl"
    y_train = pickle.load(open(y_train_file, "rb"))
    # print(f"loaded {y_train_file}")

    c = 0.1
    model = svm.LinearSVC(dual=False, class_weight="balanced", C=c)

    # Training the model
    # print("Starting SVC Training")
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    # print("Finished SVC Training")

    # saving the model
    if LOAD_SMALL_SET:
        filename = f"/scratch/owebb1/models/svm_small_{SIZE}_{FILE_NUM}.pkl"
    else:
        filename = f"/scratch/owebb1/models/svm_{SIZE}_{FILE_NUM}.pkl"

    pickle.dump(model, open(filename, "wb"))
    # print(f"Model dumped into {filename}")
    print(f"{end-start}")


if __name__ == "__main__":
    main(sys.argv[1:])
