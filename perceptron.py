import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report,plot_confusion_matrix
import matplotlib.pyplot as plt

training_data = pd.read_excel('/Users/qiaorui/Desktop/Pixel data/Assignment #1_Training dataset.xlsx')
testing_data = pd.read_excel('/Users/qiaorui/Desktop/Pixel data/Assignment #1_Testing dataset.xlsx')

x_training = training_data.iloc[0:2000,1:785]
x_testing = testing_data.iloc[0:200,1:785]
y_training = training_data.iloc[0:2000,0]
y_testing = testing_data.iloc[0:200,0]

min_max_scaler=preprocessing.MinMaxScaler()
x_training_minmax = min_max_scaler.fit_transform(x_training)
x_testing_minmax = min_max_scaler.fit_transform(x_testing)

perceptron = Perceptron (penalty=None, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
                  verbose=0, eta0=1, n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1,
                  n_iter_no_change=6, class_weight=None, warm_start=False)
y_predict = perceptron.fit(x_training_minmax,y_training).predict(x_testing_minmax)

accuracy_testing = accuracy_score(y_testing,y_predict)
precision_testing = precision_score(y_testing, y_predict, average = 'macro')
recall_testing = recall_score(y_testing, y_predict, average = 'macro')
ClassificationReport = classification_report(y_testing, y_predict, labels = [0,1,2,3,4,5,6,7,8,9])
print(ClassificationReport)
print('Accuracy = {0:0.04f}'.format(accuracy_testing),
'Precision = {0:0.04f}'.format(precision_testing),
'Recall = {0:0.04f}'.format(recall_testing))
plot_confusion_matrix(perceptron, x_testing_minmax,y_testing)
plt.show()
