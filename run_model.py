#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

from common import load_image_for_model

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "model.keras"

    model = load_model(model_path)

    image_data = load_image_for_model(image_path)

    colored = model.predict(np.array([image_data]))[0]
    plt.imshow(colored, interpolation="none")
    plt.show()
