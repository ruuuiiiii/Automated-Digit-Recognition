# Automated-Digit-Recognition

## Problem formulation
This project aims to develop an automated digit recognition model using machine learning. Given an image with an unknown digit, the model is expected to automatically recognize the digit in the image and output the corresponding number (i.e., 0 to 9).

In developing the model, one of the main goal is to compare the performance of different machine learning algorithms in supporting the automated digit recognition. Thus, this project tested the following algorithms:
* Perceptron,
* Nearest neighbor,
* Gaussian naïve Bayes
* Random forest,
* Support vector machine.

## Dataset
The MINST dataset is used in this project, which is available from https://www.kaggle.com/c/digit-recognizer/data.
A sample of digits from the dataset is shown below:

![Image of MINST](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## Performance
To compare the performance, a 3-fold cross-validation with grid search was conducted.

Support vector machine with the RBF kernel performed the best compared to the other tested algorithms.

The confusion matrix for each tested algorithm (with the best hyperparameters) is shown below.

* Figure 1. Confusion Matrix for Perceptron (Accuracy = 0.8500, Precision = 0.8546, Recall = 0.8500).
![Image of Perceptron](https://github.com/ruuuiiiii/Automated-Digit-Recognition/blob/main/Results/Perceptron.png?raw=true)

* Figure 2. Confusion Matrix for Nearest neighbor (Accuracy = 0.9250 precision = 0.9265 recall = 0.9250).
![Image of RF](https://github.com/ruuuiiiii/Automated-Digit-Recognition/blob/main/Results/KNN.png?raw=true)

* Figure 3. Confusion Matrix for Gaussian naïve Bayes (Accuracy = 0.8200 precision = 0.8363 recall = 0.8200).
![Image of RF](https://github.com/ruuuiiiii/Automated-Digit-Recognition/blob/main/Results/GNB.png?raw=true)

* Figure 4. Confusion Matrix for Random forest (Accuracy = 0.9250 Precision = 0.9265 Recall = 0.9250).
![Image of RF](https://github.com/ruuuiiiii/Automated-Digit-Recognition/blob/main/Results/RF.png?raw=true)

* Figure 5. Confusion Matrix for Support vector machine (Accuracy = 0.9650 Precision = 0.9654 Recall = 0.9650).
![Image of SVM](https://github.com/ruuuiiiii/Automated-Digit-Recognition/blob/main/Results/SVM.png?raw=true)
