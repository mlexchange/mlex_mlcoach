import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# keras callbacks for model training.  Threads while keras functions are running
# so that you can see training or evaluation of the model in progress
class TrainCustomCallback(tf.keras.callbacks.Callback):
    # For model training
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss'):
            if epoch == 0:
                print('epoch loss val_loss accuracy val_accuracy\n', flush=True)
            loss = logs.get('loss')
            val_loss = logs.get('val_loss')
            accuracy = logs.get('accuracy')
            val_accuracy = logs.get('val_accuracy')
            print(str(epoch) + ' ' + str(loss) + ' ' + str(val_loss) + ' ' + str(accuracy) + ' ' + str(val_accuracy)
                  + '\n', flush=True)
        else:
            if epoch == 0:
                print('epoch loss accuracy\n', flush=True)
            loss = logs.get('loss')
            accuracy = logs.get('accuracy')
            print(str(epoch) + ' ' + str(loss) + ' ' + str(accuracy) + '\n', flush=True)

    def on_train_end(self, logs=None):
        print('Train process completed', flush=True)


# keras callbacks for model training.  Threads while keras functions are running
# so that you can see training or evaluation of the model in progress
class TestCustomCallback(tf.keras.callbacks.Callback):
    def on_predict_begin(self, logs=None):
        print('class probability\n', flush=True)

    def on_predict_batch_end(self, batch, logs=None):
        out = logs['outputs']
        for row in range(out.shape[0]):         # when batch>1
            print(str(np.argmax(out[row,:])) + ' ' + str(np.max(out[row,:])) + '\n', flush=True)

    def on_predict_end(self, logs=None):
        print('Prediction process completed', flush=True)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
IMG_SIZE = (224, 224)       # dimensions for images: fixed due to TF models
COLOR_MODE = 'rgb'          # fixed due to TF models


# Data Augmentation + Batch Size
def data_processing(parameters, data_dir, classes, shuffle):
    rotation_angle = parameters.rotation_angle
    image_flip = parameters.image_flip
    if image_flip=='None':
        horizontal_flip = False
        vertical_flip = False
    if image_flip=='Vertical':
        horizontal_flip = False
        vertical_flip = True
    if image_flip=='Horizontal':
        horizontal_flip = True
        vertical_flip = False
    if image_flip=='Both':
        horizontal_flip = True
        vertical_flip = True
    batch_size = parameters.batch_size
    seed = 75       # fixed seed
    datagen = ImageDataGenerator(rotation_range=rotation_angle,
                                 horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip)

    first_data = glob.glob(data_dir + '/**/*.*', recursive=True)
    if len(first_data) > 0:
        data_type = os.path.splitext(first_data[0])[-1]

        if data_type in ['.tiff', '.tif', '.jpg', '.jpeg', '.png']:
            data_generator = datagen.flow_from_directory(
                data_dir,
                target_size=IMG_SIZE,
                batch_size=batch_size,
                color_mode=COLOR_MODE,
                seed=seed,
                class_mode='categorical',
                shuffle=shuffle
            )

        elif data_type == '.npy':
            data_path = first_data

            i = 0
            labels = {}
            for name in classes:
                labels[name] = i
                i += 1

            x = list()
            y = list()
            for path in data_path:
                tmp_arr = np.load(path)
                input_arr = np.array([[[0, 0, 0] for y in
                                       range(len(tmp_arr))] for x in range(len(tmp_arr))])
                for x in range(len(tmp_arr)):
                    for y in range(len(tmp_arr)):
                        rgb = int((255 * tmp_arr[x][y]) + 0.5)
                        input_arr[x][y] = [rgb, rgb, rgb]
                x.append(input_arr)
                y.append(labels[path.split('/')[-2]])
            x = np.array(x)
            y = tf.keras.utils.to_categorical(y, num_classes=len(classes))
            data_generator = datagen.flow(x=x,
                                          y=y,
                                          batch_size=batch_size,
                                          shuffle=shuffle)
        return data_generator
    return []
