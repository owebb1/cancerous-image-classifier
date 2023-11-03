# E90_Final
Owen Webb Final E90 Project


This is run in 4 stages:

1. ```python3 load_data_to_pkl.py -s -n 60 -f 0```

2. ```python3 train_test_split.py -s -n 60 -f 0```

3. ```python3 baggin_svm_train.py -s -n 60 -f 0``` or ```python3 svm_train.py -s -n 60 -f 0```

4. ```python3 model_test.py -s -n 60 -f 0```


## Repository Organization

The repository is structured as follows:

- `README.md`: The introductory document for the repository, containing the project overview, setup instructions, and usage guidelines.

- `data/`: A directory containing all the data-related files, such as CSV datasets and scripts for data handling and transformation.

- `docs/`: This folder is for storing all project-related documentation, including markdown notes and additional references.

- `models/`: Contains machine learning models and associated scripts for training and testing, including SVM with bagging and standalone SVM scripts.

- `scripts/`: Includes utility scripts for environmental setup, test automation, and other repetitive tasks.

- `images/`: Hosts images that are part of the project documentation or generated from the project's visualizations.

- `notebooks/`: If applicable, this directory will contain Jupyter notebooks used for data analysis, exploration, or experimental tracking.

- `src/`: The main source code for the project's primary functionality, including core algorithms and data processing utilities.

- `visualization/`: Dedicated to scripts for generating visual outputs from the data or model results.

