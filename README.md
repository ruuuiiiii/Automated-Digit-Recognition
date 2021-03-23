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
Support vector machine with the RBF kernel performed the best compared to the other tested algorithms. The confusion matrix for each test algorithm is shown below.
