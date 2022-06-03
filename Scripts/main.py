import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import montage as montage2d
from tensorflow import keras


import keras.backend as K
from keras.losses import binary_crossentropy


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 0.05 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)


def show_predictions(test_img_path, img_name):
    model_path = "./Models/best_model.h5"
    test_img_data = imread(os.path.join(test_img_path, img_name))
    #     test_img_data = resize(test_img_data, (300, 300, 3))
    test_out_img = [test_img_data]
    test_img = (np.stack(test_out_img, 0) / 255.0).astype(np.float32)

    seg_model = keras.models.load_model(
        model_path,
        custom_objects={
            "dice_coef": dice_coef,
            "dice_p_bce": dice_p_bce,
            "true_positive_rate": true_positive_rate,
        },
    )

    pred_y = seg_model.predict(test_img)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    montage_rgb = lambda x: np.stack(
        [montage2d(x[:, :, :, i]) for i in range(x.shape[3])], -1
    )
    ax1.imshow(montage_rgb(test_img))
    ax2.imshow(montage2d(pred_y[:, :, :, 0]), cmap="bone_r")
    ax2.set_title("Prediction")
    # fig.savefig("pred_fig.png", dpi=300)
    plt.show()

    # command to install skimage: pip install skimage


test_img_path = "./Data/test/"
# img_name = "sample_image_1.jpg"
img_name = "pcb.jpg"
# img_name = "sample_img_2_resized.JPG"
# img_name = '300x300.jpg'
show_predictions(test_img_path, img_name)
