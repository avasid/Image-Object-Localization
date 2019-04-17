import os
import pickle
import shutil

import matplotlib
import numpy as np

matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.models import load_model
import random
from keras import backend as tf


def predictions(RESOLUTION):
    def my_metric(labels, predictions):
        threshhold = 0.75
        x = predictions[:, 0] * 120
        x = tf.maximum(tf.minimum(x, 120.0), 0.0)
        y = predictions[:, 1] * 120
        y = tf.maximum(tf.minimum(y, 120.0), 0.0)
        width = predictions[:, 2] * 120
        width = tf.maximum(tf.minimum(width, 120.0), 0.0)
        height = predictions[:, 3] * 120
        height = tf.maximum(tf.minimum(height, 120.0), 0.0)
        label_x = labels[:, 0]
        label_y = labels[:, 1]
        label_width = labels[:, 2]
        label_height = labels[:, 3]
        a1 = tf.tf.multiply(width, height)
        a2 = tf.tf.multiply(label_width, label_height)
        x1 = tf.maximum(x, label_x)
        y1 = tf.maximum(y, label_y)
        x2 = tf.minimum(x + width, label_x + label_width)
        y2 = tf.minimum(y + height, label_y + label_height)
        IoU = tf.abs(tf.tf.multiply((x1 - x2), (y1 - y2))) / (a1 + a2 - tf.abs(tf.tf.multiply((x1 - x2), (y1 - y2))))
        condition = tf.less(threshhold, IoU)
        sum = tf.tf.where(condition, tf.ones(tf.shape(condition)), tf.zeros(tf.shape(condition)))
        return tf.tf.reduce_mean(sum)

    def smooth_l1_loss(true_box, pred_box):
        loss = 0.0
        for i in range(4):
            residual = tf.abs(true_box[:, i] - pred_box[:, i] * 120)
            condition = tf.less(residual, 1.0)
            small_res = 0.5 * tf.square(residual)
            large_res = residual - 0.5
            loss = loss + tf.tf.where(condition, small_res, large_res)
        return tf.tf.reduce_mean(loss)

    plt.switch_backend('agg')

    f = open("./id_to_data_test", "rb+")
    data = pickle.load(f)

    f = open("./id_to_box_test", "rb+")
    box = pickle.load(f)

    f = open("./id_to_mean_test", "rb+")
    mean = pickle.load(f)

    f = open("./id_to_std_test", "rb+")
    std = pickle.load(f)

    lenn = len(data)
    index = [i for i in range(lenn)]
    index = random.sample(index, 100)

    model = load_model('./model.h5', custom_objects={'smooth_l1_loss': smooth_l1_loss, 'my_metric': my_metric})
    result = model.predict(data[index, :, :, :])

    shutil.rmtree("./prediction/")
    os.makedirs("./prediction/")
    j = 0
    for i in index:
        print("Predicting " + str(i) + "th image.")
        true_box = box[i]
        image = data[i]
        prediction = result[j]
        j += 1
        for channel in range(3):
            meani = mean[i]
            stdi = std[i]
            image[:, :, channel] = image[:, :, channel] * stdi[channel] + meani[channel]

        image = image * 255
        image = image.astype(np.uint8)
        plt.imshow(image)

        plt.gca().add_patch(plt.Rectangle((true_box[0], true_box[1]), true_box[2], true_box[3],
                                          fill=False, edgecolor='red', linewidth=2, alpha=0.5))
        plt.gca().add_patch(plt.Rectangle((prediction[0] * RESOLUTION, prediction[1] * RESOLUTION),
                                          prediction[2] * RESOLUTION, prediction[3] * RESOLUTION,
                                          fill=False, edgecolor='green', linewidth=2, alpha=0.5))
        plt.show()
        plt.savefig("./prediction/" + str(i) + ".png")
        plt.cla()
