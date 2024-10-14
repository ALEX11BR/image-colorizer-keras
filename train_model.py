#!/usr/bin/env python3

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Reshape, Concatenate, Input, Flatten
from tensorflow.keras.models import Model, load_model

from common import IMAGE_LEN

FILTERS_COUNTS = [32, 64, 128]

def load_images(images_dir):
    if os.path.isfile(os.path.join(images_dir, "images.pkl")):
        with open(os.path.join(images_dir, "images.pkl"), "rb") as pkl_file:
            x_images, y_images = pickle.load(pkl_file)
    else:
        x_images = []
        y_images = []

        for image_file in os.listdir(images_dir):
            try:
                image_data = Image.open(os.path.join(images_dir, image_file)).resize((IMAGE_LEN, IMAGE_LEN))
                image_data = ImageOps.exif_transpose(image_data)

                y_images.append(np.array(image_data)[...,:3] / 255)
                x_images.append(np.array(image_data.convert("L")) / 255)

            except Exception as e:
                print("A file read failed:", e, file=sys.stderr)

        x_images = np.array(x_images).reshape(len(x_images), IMAGE_LEN, IMAGE_LEN, 1)
        y_images = np.array(y_images).reshape(len(y_images), IMAGE_LEN, IMAGE_LEN, 3)
        with open(os.path.join(images_dir, "images.pkl"), "wb") as pkl_file:
            pickle.dump((x_images, y_images), pkl_file)

    return train_test_split(x_images, y_images, test_size=0.2)

def build_colorization_model():
    # encoder part; ending with a latent vector
    input = Input(shape=(IMAGE_LEN, IMAGE_LEN, 1))
    x = input

    layers_to_concat = []
    for i in range(len(FILTERS_COUNTS)):
        x = Conv2D(filters=FILTERS_COUNTS[i],
                   kernel_size=3,
                   strides=2,
                   activation="relu",
                   padding="same")(x)
        if i < len(FILTERS_COUNTS) - 1:
            layers_to_concat.append(x)
    encoder_out_shape = K.int_shape(x)[1:]
    x = Flatten()(x)
    x = Dense(256)(x)

    # decoder part
    x = Dense(encoder_out_shape[0] * encoder_out_shape[1] * encoder_out_shape[2])(x)
    x = Reshape(encoder_out_shape)(x)
    for i in range(len(FILTERS_COUNTS) - 1, -1, -1):
        if i < len(FILTERS_COUNTS) - 1:
            x = Concatenate(axis=3)([x, layers_to_concat[i]])
        x = Conv2DTranspose(filters=FILTERS_COUNTS[i],
                            kernel_size=3,
                            strides=2,
                            activation="relu",
                            padding="same")(x)
    x = Concatenate(axis=3)([x, input])
    x = Conv2DTranspose(filters=3,
                        kernel_size=3,
                        activation="sigmoid",
                        padding="same")(x)

    model = Model(inputs=input, outputs=x)
    return model

if __name__ == "__main__":
    images_dir = sys.argv[1] if len(sys.argv) > 1 else "images"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "model.keras"

    x_train, x_test, y_train, y_test = load_images(images_dir)

    try:
        model = load_model(model_path)
    except:
        model = build_colorization_model()

    callback_checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor="loss",
                                 verbose=1,
                                 save_best_only=True,
                                 save_freq=10)

    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    fit_history = model.fit(x_train, y_train,
                            epochs=int(os.getenv("TRAIN_EPOCHS") or "100"),
                            validation_data=(x_test, y_test),
                            callbacks=[callback_checkpoint])
    model.save(model_path)

    plt.plot(fit_history.history["loss"])
    plt.title("Loss")

    y_test_predicted = model.predict(x_test)

    fig = plt.figure()
    plt.axis("off")
    plt.title("Grayscale images")
    for i in range(9):
        fig.add_subplot(3, 3, i + 1)
        plt.axis("off")
        plt.imshow(x_test[i], cmap="gray")

    fig = plt.figure()
    plt.axis("off")
    plt.title("Colorized images")
    for i in range(9):
        fig.add_subplot(3, 3, i + 1)
        plt.axis("off")
        plt.imshow(y_test_predicted[i], interpolation="none")

    fig = plt.figure()
    plt.axis("off")
    plt.title("Proper images")
    for i in range(9):
        fig.add_subplot(3, 3, i + 1)
        plt.axis("off")
        plt.imshow(y_test[i], interpolation="none")

    plt.show()
