import glob
import os

import numpy as np
from scipy.ndimage import interpolation
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
    def __init__(self, filenames=None, classes=None):
        self.classes = classes
        self.filenames = filenames

    def on_predict_begin(self, logs=None):
        print('Prediction process started\n', flush=True)

    def on_predict_batch_end(self, batch, logs=None):
        out = logs['outputs']
        batch_size = out.shape[0]
        if batch==0:
            msg = ['filename'] + self.classes
            print(' '.join(msg) + '\n', flush=True)
        filenames = self.filenames[batch*batch_size:(batch+1)*batch_size]
        for row in range(batch_size):         # when batch>1
            prob = np.char.mod('%f', out[row,:])
            print(filenames[row]+ ' ' + ' '.join(prob) + '\n', flush=True)

    def on_predict_end(self, logs=None):
        print('Prediction process completed', flush=True)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
IMG_SIZE = (224, 224)       # dimensions for images: fixed due to TF models
COLOR_MODE = 'rgb'          # fixed due to TF models


# Data Augmentation + Batch Size
def data_processing(parameters, data_dir):
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
    if parameters.target_width and parameters.target_height:
        target_width = parameters.target_width
        target_height = parameters.target_height
    else:
        target_width = 224
        target_height = 224
    if parameters.shuffle:
       shuffle = parameters.shuffle
    else:
        shuffle = False
    if parameters.seed:
        seed = parameters.seed
    else:
        seed = 45       # fixed seed
    if parameters.val_pct:
        datagen = ImageDataGenerator(rotation_range=rotation_angle,
                                     horizontal_flip=horizontal_flip,
                                     vertical_flip=vertical_flip,
                                     validation_split=parameters.val_pct/100)
    else:
        datagen = ImageDataGenerator(rotation_range=rotation_angle,
                                     horizontal_flip=horizontal_flip,
                                     vertical_flip=vertical_flip)

    data_generator = []
    first_data = glob.glob(data_dir + '/**/*.*', recursive=True)
    if len(first_data) > 0:
        data_type = os.path.splitext(first_data[0])[-1]
        if data_type in ['.tiff', '.tif', '.jpg', '.jpeg', '.png']:
            if parameters.val_pct:
                train_generator = datagen.flow_from_directory(data_dir,
                                                              target_size=(target_width, target_height),
                                                              color_mode=COLOR_MODE,
                                                              class_mode='categorical',
                                                              batch_size=batch_size,
                                                              shuffle=shuffle,
                                                              seed=seed,
                                                              subset='training')
                valid_generator = datagen.flow_from_directory(data_dir,
                                                              target_size=(target_width, target_height),
                                                              color_mode=COLOR_MODE,
                                                              class_mode='categorical',
                                                              batch_size=batch_size,
                                                              shuffle=shuffle,
                                                              seed=seed,
                                                              subset='validation')
            else:
                train_generator = datagen.flow_from_directory(data_dir,
                                                              target_size=(target_width, target_height),
                                                              color_mode=COLOR_MODE,
                                                              class_mode='categorical',
                                                              batch_size=batch_size,
                                                              shuffle=shuffle,
                                                              seed=seed)
                valid_generator = []
            data_generator = (train_generator, valid_generator)
    
    if os.path.splitext(data_dir)[-1] == '.npz':
        x_key = parameters.x_key
        y_key = parameters.y_key
        data = np.load(data_dir)
        x_data = data[x_key]
        y_data = data[y_key]
        y_data = tf.keras.utils.to_categorical(y_data, num_classes=len(np.unique(y_data)))
        if target_width != x_data.shape[1] or target_height != x_data.shape[2]: # resize if needed
            data_shape = list(x_data.shape)
            data_shape[0] = 1
            data_shape[1] = target_width/data_shape[1]
            data_shape[2] = target_height/data_shape[2]
            x_data = interpolation.zoom(x_data, data_shape)
            if len(data_shape)==3:
                x_data = np.repeat(x_data[:,:,:,np.newaxis], 3, axis=3)  # RGB
        print(y_data.shape)
        if parameters.val_pct:
            train_generator = datagen.flow(x=x_data,
                                           y=y_data,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           seed=seed,
                                           subset='training')
            valid_generator = datagen.flow(x=x_data,
                                           y=y_data,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           seed=seed,
                                           subset='validation')
        else:
            train_generator = datagen.flow(x=x_data,
                                           y=data[y_key],
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           seed=seed)
            valid_generator = []
        data_generator = (train_generator, valid_generator)
        # elif data_type == '.npy':         # not fully supported at this time
        #     data_path = first_data
        #
        #     i = 0
        #     labels = {}
        #     for name in classes:
        #         labels[name] = i
        #         i += 1
        #
        #     x = list()
        #     y = list()
        #     for path in data_path:
        #         tmp_arr = np.load(path)
        #         input_arr = np.array([[[0, 0, 0] for y in
        #                                range(len(tmp_arr))] for x in range(len(tmp_arr))])
        #         for x in range(len(tmp_arr)):
        #             for y in range(len(tmp_arr)):
        #                 rgb = int((255 * tmp_arr[x][y]) + 0.5)
        #                 input_arr[x][y] = [rgb, rgb, rgb]
        #         x.append(input_arr)
        #         y.append(labels[path.split('/')[-2]])
        #     x = np.array(x)
        #     y = tf.keras.utils.to_categorical(y, num_classes=len(classes))
        #     data_generator = datagen.flow(x=x,
        #                                   y=y,
        #                                   batch_size=batch_size,
        #                                   shuffle=shuffle)
    return data_generator
