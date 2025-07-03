from PIL import Image
import numpy as np

def prepare_image(image):
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)