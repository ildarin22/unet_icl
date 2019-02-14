from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam


def build_model(in_w, in_h, classes):

    in_tensor = Input(shape=(in_w, in_h, 3))
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=in_tensor,
        input_shape = (in_w, in_h, 3),
        pooling='avg'
    )

    for layer in base_model.layers:
        layer.trainable = False

    op = Dense(256, activation='relu')(base_model.output)
    op = Dropout(.25)(op)

    out_tensor = Dense(classes, activation='softmax')(op)
    model = Model(inputs=in_tensor, outputs=out_tensor)
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model
