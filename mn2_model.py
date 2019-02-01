from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Input, Dropout
from keras.models import Model

def build_model(in_w, in_h):
    in_tensor = Input(shape=(in_w, in_h, 3))
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=in_tensor,
        input_shape = (in_w, in_h, 3),
        pooling='avg'
    )

    op = Dense(256, activation='relu')(base_model.output)
    op = Dropout(.25)(op)

    out_tensor = Dense(10, activation='softmax')(op)
    model = Model(inputs=in_tensor, outputs=out_tensor)

    return model
