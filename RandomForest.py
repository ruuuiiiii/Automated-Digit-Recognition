import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

testing_data = pd.read_excel('/Users/qiaorui/Desktop/Pixel data/Assignment #1_Testing dataset.xlsx')
training_data = pd.read_excel('/Users/qiaorui/Desktop/Pixel data/Assignment #1_Training dataset.xlsx')

x_training = training_data.iloc[0:2000,1:785]
y_training = training_data.iloc[0:2000,0]
x_testing = testing_data.iloc[0:200,1:785]
y_testing = testing_data.iloc[0:200,0]

min_max_scaler = preprocessing.MinMaxScaler()
x_training_minmax = min_max_scaler.fit_transform(x_training)
x_testing_minmax = min_max_scaler.fit_transform(x_testing)

forest = RandomForestClassifier()
y_predict = forest.fit(x_training_minmax,y_training).predict(x_testing_minmax)

accuracy_testing = accuracy_score(y_testing,y_predict)
precision_testing = precision_score(y_testing, y_predict, average = 'macro')
recall_testing = recall_score(y_testing, y_predict, average = 'macro')
ClassificationReport = classification_report(y_testing, y_predict, labels = [0,1,2,3,4,5,6,7,8,9])
print(ClassificationReport)
print('Accuracy = {0:0.04f}'.format(accuracy_testing),
'Precision = {0:0.04f}'.format(precision_testing),
'Recall = {0:0.04f}'.format(recall_testing))
plot_confusion_matrix(forest, x_testing_minmax,y_testing)
plt.show()
