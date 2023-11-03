import pickle
import warnings
import sys
import getopt
import os
import time

from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


FILE_DIR = os.listdir("/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/")
BASE_PATH = "/scratch/owebb1/archive-2/IDC_regular_ps50_idx5/"
PICKLE_PATH = "/scratch/owebb1/"
FOLDER = os.listdir(BASE_PATH)
LOAD_SMALL_SET = False


def set_args(args):
    global LOAD_SMALL_SET
    try:
        opts, _ = getopt.getopt(args, "hs", [])
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
            LOAD_SMALL_SET = True


# DEFINITION: Main will load the images in if flagged
def main(argv):
    set_args(argv)

    # determine whether to use big or small dataset
    if LOAD_SMALL_SET == True:
        file_suffix = "small"
    else:
        file_suffix = "big"

    # open the training data
    X_train_file = f"/scratch/owebb1/X_train_{file_suffix}.pkl"
    X_train = pickle.load(open(X_train_file, "rb"))
    print(f"loaded {X_train_file}")
    y_train_file = f"/scratch/owebb1/y_train_{file_suffix}.pkl"
    y_train = pickle.load(open(y_train_file, "rb"))
    print(f"loaded {y_train_file}")

    c = 0.1
    svc = svm.SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        C=c,
    )  # , kernel=kernal)

    model = AdaBoostClassifier(
        base_estimator=svc, n_estimators=40  # , n_jobs=-1, max_samples=4400
    )

    # Training the model
    print("Starting SVC Training")
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print("Finished SVC Training")

    # saving the model
    filename = f"/scratch/owebb1/boosting_svm_small.pkl"
    pickle.dump(model, open(filename, "wb"))
    print(f"Model dumped into {filename}")
    print(f"Training step took {end-start}s")


if __name__ == "__main__":
    main(sys.argv[1:])
