# Breast Cancer Histopathology Image Classification with SVM

## Abstract

Breast cancer is a leading type of cancer in women, and any advancements that can aid in early detection are invaluable. This project leverages the capabilities of machine learning to assist pathologists in the early detection of cancer within tissue samples. Our focus is on employing Support Vector Machines (SVM) to classify histopathology images of breast tissue as either cancerous or non-cancerous.

Using SVM implementations from `sklearn` and image data from Cruz-Roa et al, we evaluated the efficacy of a single Support Vector Classifier and a Bagging classifier with eight SVMs as base models. The single SVM achieved an accuracy of 71.0%, while the Bagging classifier reached 76.1% accuracy on 22% of the dataset. These results do not match the accuracy of CNNs from prior studies, which is approximately 84%. It suggests that SVM may not be the optimal choice for predicting breast cancer in this context.

For further details, refer to the full paper: ["Machine Learning for Breast Cancer Histopathology Image Classification"](http://hdl.handle.net/10066/24409).

## Project Structure

- `README.md`: Introduction and project summary.
- `data/`: Contains the datasets used in this study, including the histopathology images and their labels.
- `docs/`: Additional project documentation and notes.
- `models/`:
  - `svm_train.py`: Script for training the SVM classifier.
  - `bagging_svm_train.py`: Script for training the Bagging classifier with SVM base models.
  - `model_test.py`: Script for evaluating the models' performance.
- `scripts/`: Utility scripts for setting up and managing the project.
- `images/`: Visualization of data and model outputs.
- `src/`:
  - `load_data_to_pkl.py`: Script to load and preprocess the data.
  - `train_test_split.py`: Script for splitting the data into training and testing sets.
- `visualization/`:
  - `visualize.py`: Script for generating and saving visual results.

## Usage

To replicate the study or to use the models for your own research, follow these steps:

1. Prepare the data:

This is run in 4 stages:

1. `python3 load_data_to_pkl.py -s -n 60 -f 0`

2. `python3 train_test_split.py -s -n 60 -f 0`

3. `python3 baggin_svm_train.py -s -n 60 -f 0` or `python3 svm_train.py -s -n 60 -f 0`

4. `python3 model_test.py -s -n 60 -f 0`

## Contributing

Contributions to this project are welcome. To contribute, please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Cruz-Roa et al for providing the histopathology image dataset.
- sklearn for their Support Vector Machine implementation.
