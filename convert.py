"""
A sample of convert the cifar100 dataset to 224 * 224 size train\val data.
"""
import cv2
import os
from keras.datasets import cifar10


def convert():
    train = 'data//train//'
    val = 'data//validation//'


    (X_train, y_train), (X_test, y_test) = cifar10.load_data() # load_data(label_mode='fine')

    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]
        path = train + str(y[0])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_CUBIC)
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + '//' + str(i) + '.jpg', x)

    for i in range(len(X_test)):
        # Changed due to issue https://github.com/xiaochus/MobileNetV2/issues/3
        x = X_test[i]
        y = y_test[i]
        path = val + str(y[0])
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_CUBIC)
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + '//' + str(i) + '.jpg', x)


if __name__ == '__main__':
    convert()
