import numpy as np
from PIL import Image, ImageOps

IMAGE_LEN = 200

def load_image_for_model(path):
    image = Image.open(path).resize((IMAGE_LEN, IMAGE_LEN)).convert("L")
    image = ImageOps.exif_transpose(image)
    return np.array(image) / 255
