import cytoolz as toolz
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Model


def main():
    mnist = fetch_mldata('MNIST original')
    X = mnist.data.astype(np.float32).reshape((len(mnist.data), 28, 28, 1)) / 255.
    label_binarizer = LabelBinarizer()
    Y = label_binarizer.fit_transform(mnist.target)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=60000)

    layers = [
        Convolution2D(20, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(),
        Convolution2D(50, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(500, activation='relu'),
        Dense(10, activation='softmax')
    ]

    input = Input((28, 28, 1))
    output = toolz.pipe(input, *layers)
    model = Model(input, output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=128, validation_data=[X_test, Y_test])


if __name__ == '__main__':
    main()
