# generic tools
import numpy as np

# tools from sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

# matplotlib
import matplotlib.pyplot as plt

def download_data():
    data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    data = data.astype("float")/255.0
    (X_train, X_test, y_train, y_test) = train_test_split(data,
                                                        labels, 
                                                        test_size=0.2)
    return X_train, X_test, y_train, y_test

def convert_data(y_train, y_test):
    lb = LabelBinarizer()
    y_train_binarized = lb.fit_transform(y_train)
    y_test_binarized = lb.fit_transform(y_test)
    return y_train_binarized, y_test_binarized

def model():
    compiled_model = Sequential()
    compiled_model.add(Dense(256, 
                    input_shape=(784,), 
                    activation="relu"))
    compiled_model.add(Dense(128, 
                    activation="relu"))
    compiled_model.add(Dense(10, 
                    activation="softmax"))
    sgd = SGD(learning_rate = 0.01)
    compiled_model.compile(loss="categorical_crossentropy", 
                optimizer=sgd, 
                metrics=["accuracy"])
    return compiled_model

def train_model(compiled_model, X_train, y_train_binarized):
    train_val = compiled_model.fit(X_train, y_train_binarized, 
                    validation_split=0.1,
                    epochs=2, 
                    batch_size=32)
    return train_val

def plot(train_val):
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, 10), train_val.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 10), train_val.history["val_loss"], label="val_loss", linestyle=":")
    plt.plot(np.arange(0, 10), train_val.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 10), train_val.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()

def main():
    X_train, X_test, y_train, y_test = download_data()
    y_train_binarized, y_test_binarized = convert_data(y_train, y_test)
    compiled_model = model()
    train_val = train_model(compiled_model, X_train, y_train_binarized)
    plot(train_val)

if __name__== "__main__":
    main()