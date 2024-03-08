import os
import sys
sys.path.append("..")
import cv2

# Import teaching utils
import numpy as np
from imutils import jimshow as show 
from imutils import jimshow_channel as show_channel

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import matplotlib.pyplot as plt

# Import dataset
from tensorflow.keras.datasets import cifar10


def load_data():
    return cifar10.load_data()

def reshape_data():
    X_list_train = []

    for image in X_train:
    X_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    X_normalized = X_grey/255
    X_list_train.append(X_normalized)

    X_train_final = np.array(X_list).reshape(-1, 1024)

    X_list_test = []

    for image in X_test:
        X_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X_normalized = X_grey/255
        X_list_test.append(X_normalized)

    X_test_final = np.array(X_list).reshape(-1, 1024)
    return X_test_final, X_train_final

def classifier(X_test_final, X_train_final):
    classifierLogistic = LogisticRegression(tol=0.1, 
                         solver='saga',
                         multi_class='multinomial').fit(X_train_final, y_train)
    y_pred = classifierLogistic.predict(X_test_final)
    classifier_metrics_logistic = metrics.classification_report(y_test, y_pred, target_names= labels)
    print(classifier_metrics_logistic)
    return classifier_metrics_logistic

def file_save(classifier_metrics_logistic):
    text_file = open("../out/neuralnetwork.txt", 'w')
    text_file.write(classifier_metrics_logistic)
    text_file.close()


def main():
    (X_train, y_train), (X_test, y_test) = load_data()
    x_test = reshape_data()
    classifier(x_test)

if __name__=="__main__":
    main()