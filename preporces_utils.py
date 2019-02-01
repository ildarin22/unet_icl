import numpy as np
from glob import glob
from PIL import Image
import os

SIZE = 96

train_images =glob("D:\\ds\\whiskey\\whisky_labels\\train\\**\\*.jpg", recursive=True)
test_images = glob("D:\\ds\\whiskey\\whisky_labels\\test\\**\\*.jpg", recursive=True)


whisky_labels = next(os.walk('D:\\ds\\whiskey\\whisky_labels\\train\\'))[1]
dights_labels = dict(zip(range(10), whisky_labels))
d = {}

for k, v in dict(zip(range(10), whisky_labels)).items():
    d[k] = glob("D:\\ds\\whiskey\\whisky_labels\\train\\"+v+"\\*.jpg")


def import_image( filename):
    img = Image.open(filename).convert("LA").resize( (SIZE,SIZE))
    return np.array(img)[:,:,0]

train_img = np.array([import_image(img) for img in train_images])
x_whiskey_train = train_img
