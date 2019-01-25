
import numpy as np
from keras.callbacks import ModelCheckpoint
from segmentation_models import Unet, PSPNet
from segmentation_models.backbones import get_preprocessing
from sklearn.model_selection import train_test_split
from unet_seg.plot import plotting, plot_sample


from unet_seg.data_load import get_data
import os

root_path = os.getcwd()
path_train = root_path+'\\input\\train\\'
path_test = root_path+'\\input\\test\\'
# backbone = 'resnet34'
backbone = 'resnet50'
checkpoint = 'model-bottles_'+backbone+'.h5'
input_shape = (256, 256, 1)

X, y = get_data(path_train, input_shape, train=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)


def unet_train():
    callbacks = [
            # EarlyStopping(patience=10, verbose=1),
            # ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
            ModelCheckpoint('unet_'+checkpoint, verbose=1, monitor='loss', save_best_only=True, save_weights_only=True)
        ]
    model = Unet(backbone_name=backbone, encoder_weights=None, input_shape=input_shape)
    model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])
    results = model.fit(X_train, y_train, callbacks=callbacks, epochs=100, verbose=1, validation_data=(X_valid,y_valid))
    plotting(results)



def net_predict():
    model = Unet(backbone_name=backbone, encoder_weights=None, input_shape=(256, 256, 1))
    model.load_weights(checkpoint)
    preds_train = model.predict(X_train, verbose=1)
    preds_val = model.predict(X_valid, verbose=1)

    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=None)

# net_predict()
unet_train()
