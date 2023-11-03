# E90_Final
Owen Webb Final E90 Project


This is run in 4 stages:

1. ```python3 load_data_to_pkl.py -s -n 60 -f 0```

2. ```python3 train_test_split.py -s -n 60 -f 0```

3. ```python3 baggin_svm_train.py -s -n 60 -f 0``` or ```python3 svm_train.py -s -n 60 -f 0```

4. ```python3 model_test.py -s -n 60 -f 0```


## Here is the organzation of how this is laid out

- `README.md`: This should be the entry point for anyone visiting your repository. It should contain an introduction to your project, how to set it up, and how to use it.
- `data/`: Contains all data-related files, including datasets in CSV format and scripts for handling data.
- `docs/`: Here, you can store your project notes and any additional documentation.
- `models/`: This directory should contain all your machine learning models, training scripts, and testing scripts.
- `scripts/`: Contains utility scripts that might be used for setting up the environment, running tests, or other repetitive tasks.
- `images/`: If you have images related to the project, such as plots or diagrams, they go here.
- `notebooks/`: If you're using Jupyter notebooks for analysis or data exploration, they should be placed in this directory.
- `src/`: The main source code for your project. This might include core algorithms, data processing, etc.
- `visualization/`: Dedicated to scripts that generate visual outputs from your data or model results.
