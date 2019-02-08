import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from skimage.transform import resize
import os


# SIZE = 96
# train_images =glob("D:\\ds\\whiskey\\whisky_labels\\train\\**\\*.jpg", recursive=True)
#и
# whisky_labels = next(os.walk('D:\\ds\\whiskey\\whisky_labels\\train\\'))[1]
# y_train = np.empty(0)


def preprocess_image(x, SIZE):
    # Resize the image to have the shape of (96,96)
    x = resize(x, (SIZE, SIZE),
               mode='constant',
               anti_aliasing=False)

    # convert to 3 channel (RGB)
    x = np.stack((x,) * 3, axis=-1)

    # Make sure it is a float32, here is why
    # https://www.quora.com/When-should-I-use-tf-float32-vs-tf-float64-in-TensorFlow
    return x.astype(np.float32)


def import_image(filename, SIZE):
    img = Image.open(filename).convert("LA").resize((SIZE, SIZE))
    return np.array(img)[:, :, 0]


def load_data_generator(x, y, SIZE, batch_size=30):
    num_samples = x.shape[0]
    while 1:  # Loop forever so the generator never terminates
        try:
            shuffle(x)
            for i in range(0, num_samples, batch_size):
                x_data = [preprocess_image(im, SIZE) for im in x[i:i + batch_size]]
                y_data = y[i:i + batch_size]

                # convert to numpy array since this what keras required
                yield shuffle(np.array(x_data), np.array(y_data))
        except Exception as err:
            print(err)


def get_labels(whisky_labels, path):
    y = np.empty(0)
    for k, v in dict(zip(range(10), whisky_labels)).items():
        y = np.append(y, np.full(len(next(os.walk(path))), k))
    return y
