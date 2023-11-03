from json import load
import pickle
from re import L
import warnings
import sys
import getopt
import os
from sklearn.metrics import confusion_matrix


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


def main(argv):
    set_args(argv)

    if LOAD_SMALL_SET == True:
        file_suffix = f"small_{SIZE}_{FILE_NUM}"
    else:
        file_suffix = "big"

    X_test_file = f"/scratch/owebb1/X_and_Y_test/X_test_{file_suffix}.pkl"
    X_test = pickle.load(open(X_test_file, "rb"))
    y_test_file = f"/scratch/owebb1/X_and_Y_test/y_test_{file_suffix}.pkl"
    y_test = pickle.load(open(y_test_file, "rb"))

    # Load in the pretrained model
    filename = f"/scratch/owebb1/models/bagging_svm_small_20_2.pkl"
    loaded_model = pickle.load(open(filename, "rb"))
    # print(f"loaded {filename}")

    score = loaded_model.score(X_test, y_test)
    y_pred = loaded_model.predict(X_test)
    rounded = round(score, 3)

    cm = confusion_matrix(y_test, y_pred)

    print(f"{SIZE},{FILE_NUM},{rounded}")
    # print()
    # print(f"{cm}")
    # print()


if __name__ == "__main__":
    main(sys.argv[1:])
