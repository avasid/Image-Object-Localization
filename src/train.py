from keras import backend as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, add
from keras.layers import MaxPooling2D, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from utils import plot_model, getdata


def training(RESOLUTION):
    data_train, box_train, data_test, box_test = getdata()

    # metric function
    def my_metric(labels, predictions):
        threshold = 0.75
        x = predictions[:, 0] * RESOLUTION
        x = tf.maximum(tf.minimum(x, RESOLUTION), 0.0)
        y = predictions[:, 1] * RESOLUTION
        y = tf.maximum(tf.minimum(y, RESOLUTION), 0.0)
        width = predictions[:, 2] * RESOLUTION
        width = tf.maximum(tf.minimum(width, RESOLUTION), 0.0)
        height = predictions[:, 3] * RESOLUTION
        height = tf.maximum(tf.minimum(height, RESOLUTION), 0.0)
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
        condition = tf.less(threshold, IoU)
        sum = tf.tf.where(condition, tf.ones(tf.shape(condition)), tf.zeros(tf.shape(condition)))
        return tf.tf.reduce_mean(sum)

    # loss function
    def smooth_l1_loss(true_box, pred_box):
        loss = 0.0
        for i in range(4):
            residual = tf.abs(true_box[:, i] - pred_box[:, i] * RESOLUTION)
            condition = tf.less(residual, 1.0)
            small_res = 0.5 * tf.square(residual)
            large_res = residual - 0.5
            loss = loss + tf.tf.where(condition, small_res, large_res)
        return tf.tf.reduce_mean(loss)

    def resnet_block(inputs, num_filters, kernel_size, strides, activation='relu'):
        x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(inputs)
        x = BatchNormalization()(x)
        if (activation):
            x = Activation('relu')(x)
        return x

    def resnet18():
        inputs = Input((RESOLUTION, RESOLUTION, 3))

        # conv1
        x = resnet_block(inputs, 64, [7, 7], 2)

        # conv2
        x = MaxPooling2D([3, 3], 2, 'same')(x)
        for i in range(2):
            a = resnet_block(x, 64, [3, 3], 1)
            b = resnet_block(a, 64, [3, 3], 1, activation=None)
            x = add([x, b])
            x = Activation('relu')(x)

        # conv3
        a = resnet_block(x, 128, [1, 1], 2)
        b = resnet_block(a, 128, [3, 3], 1, activation=None)
        x = Conv2D(128, kernel_size=[1, 1], strides=2, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-3))(x)
        x = add([x, b])
        x = Activation('relu')(x)

        a = resnet_block(x, 128, [3, 3], 1)
        b = resnet_block(a, 128, [3, 3], 1, activation=None)
        x = add([x, b])
        x = Activation('relu')(x)

        # conv4
        a = resnet_block(x, 256, [1, 1], 2)
        b = resnet_block(a, 256, [3, 3], 1, activation=None)
        x = Conv2D(256, kernel_size=[1, 1], strides=2, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-3))(x)
        x = add([x, b])
        x = Activation('relu')(x)

        a = resnet_block(x, 256, [3, 3], 1)
        b = resnet_block(a, 256, [3, 3], 1, activation=None)
        x = add([x, b])
        x = Activation('relu')(x)

        # conv5
        a = resnet_block(x, 512, [1, 1], 2)
        b = resnet_block(a, 512, [3, 3], 1, activation=None)
        x = Conv2D(512, kernel_size=[1, 1], strides=2, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-3))(x)
        x = add([x, b])
        x = Activation('relu')(x)

        a = resnet_block(x, 512, [3, 3], 1)
        b = resnet_block(a, 512, [3, 3], 1, activation=None)
        x = add([x, b])
        x = Activation('relu')(x)
        y = GlobalAveragePooling2D(data_format="channels_last")(x)

        # out:512
        y = Dense(1000, kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(y)
        outputs = Dense(4, kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(y)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    model = resnet18()

    model.compile(loss=smooth_l1_loss, optimizer=Adam(), metrics=[my_metric])

    model.summary()

    def lr_sch(epoch):
        # 200 total
        if epoch < 50:
            return 1e-3
        if 50 <= epoch < 100:
            return 1e-4
        if epoch >= 100:
            return 1e-5

    lr_scheduler = LearningRateScheduler(lr_sch)
    lr_reducer = ReduceLROnPlateau(monitor='val_my_metric', factor=0.2, patience=5, mode='max', min_lr=1e-3)

    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    model_details = model.fit(data_train, box_train, batch_size=128, epochs=100, shuffle=True, validation_split=0.1,
                              callbacks=[lr_scheduler, lr_reducer, checkpoint], verbose=1)

    model.save('model.h5')

    scores = model.evaluate(data_test, box_test, verbose=1)
    print('Test loss : ', scores[0])
    print('Test accuracy : ', scores[1])

    plot_model(model_details)
    return model
