import glob
import os
import math

import pandas as pd
import numpy as np
import requests
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
SPLASH_CLIENT = 'http://splash:80/api/v0'


# Get dataset from splash-ml
def load_from_splash(uri_list):
    '''
    This function queries labels from splash-ml.
    Args:
        uri_list:    URI of dataset (e.g. file path)
    Returns:
        splash_df:   Dataframe of labeled images (docker path)
    '''
    url = f'{SPLASH_CLIENT}/datasets?'
    try:
        params = {'uris': uri_list, 'page[limit]': 1000}
        datasets = requests.get(url, params=params).json()
    except Exception as e:
        print(f'Loading from splash exception: {e}')
        datasets = []
        for i in range(math.ceil(len(uri_list)/25)):
            params = {'uris': uri_list[i*25:min(25*(i+1), len(uri_list))], 'page[limit]':1000}
            datasets = datasets + requests.get(url, params=params).json()
    labels_name_data = []
    for dataset in datasets:
        for tag in dataset['tags']:
            if tag['name'] == 'labelmaker':
                labels_name_data.append([dataset['uri'], tag['locator']['path']])
    splash_df = pd.DataFrame(data=labels_name_data, 
                             index=None, 
                             columns=['filename', 'class'])
    classes = list(splash_df['class'].unique())
    return splash_df, classes


# Data Augmentation + Batch Size
def data_processing(parameters, data_dir, no_label=False):
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
    uri_list = parameters.splash
    if parameters.val_pct:
        datagen = ImageDataGenerator(rotation_range=rotation_angle,
                                     rescale=1/255,
                                     horizontal_flip=horizontal_flip,
                                     vertical_flip=vertical_flip,
                                     validation_split=parameters.val_pct/100)
    else:
        datagen = ImageDataGenerator(rotation_range=rotation_angle,
                                     rescale=1/255,
                                     horizontal_flip=horizontal_flip,
                                     vertical_flip=vertical_flip)

    data_generator = []
    if not(uri_list):
        first_data = glob.glob(data_dir + '/**/*.*', recursive=True)
        if len(first_data) > 0:
            data_type = os.path.splitext(first_data[0])[-1]
            if data_type in ['.tiff', '.tif', '.jpg', '.jpeg', '.png']:
                if no_label:
                    classes = None
                    datagen = ImageDataGenerator(rescale=1/255)
                    
                    list_filename = []
                    for dirpath, subdirs, files in os.walk(data_dir):
                        for file in files:
                            if os.path.splitext(file)[-1] in ['.tiff', '.tif', '.jpg', '.jpeg', '.png'] and not ('.' in os.path.splitext(file)[0]):
                                filename = os.path.join(dirpath, file)
                                list_filename.append(filename)
                    
                    data_df = pd.DataFrame(data=list_filename,
                                           index=None,
                                           columns=['filename'])
                    train_generator = datagen.flow_from_dataframe(data_df,
                                                        directory=data_dir,
                                                        x_col='filename',
                                                        target_size=(target_width, target_height),
                                                        color_mode=COLOR_MODE,
                                                        class_mode=None,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle,
                                                        seed=seed,
                                                        )
                    #train_generator = datagen.flow_from_directory('/'.join(data_dir.split('/')[0:-1]),
                    #                                              target_size=(target_width, target_height),
                    #                                              color_mode=COLOR_MODE,
                    #                                              class_mode=None,
                    #                                              batch_size=batch_size,
                    #                                              shuffle=shuffle,
                    #                                              seed=seed)
                    valid_generator = []
                elif parameters.val_pct:
                    classes = [subdir for subdir in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, subdir))]
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
                    classes = [subdir for subdir in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, subdir))]
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
                w = target_width/data_shape[1]
                h = target_height/data_shape[2]
                for channel in data_shape[0]:
                    x_data[channel,:] = interpolation.zoom(x_data[channel,:], [1,w,h])
                if data_shape[0]==1:
                    x_data = np.repeat(x_data[:,:,:,np.newaxis], 3, axis=3)         # RGB
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
            classes = np.unique(train_generator.__dict__['y'], axis=0)
            data_generator = (train_generator, valid_generator)
    
    else:
        splash_df, classes = load_from_splash(uri_list)
        #print(splash_df)
        if parameters.val_pct:
            train_generator = datagen.flow_from_dataframe(splash_df,
                                                        directory=data_dir,
                                                        x_col='filename',
                                                        y_col='class',
                                                        target_size=(target_width, target_height),
                                                        color_mode=COLOR_MODE,
                                                        classes=None,
                                                        class_mode='categorical',
                                                        batch_size=batch_size,
                                                        shuffle=shuffle,
                                                        seed=seed,
                                                        subset='training',
                                                        )
            
            valid_generator = datagen.flow_from_dataframe(splash_df,
                                                        directory=data_dir,
                                                        x_col='filename',
                                                        y_col='class',
                                                        target_size=(target_width, target_height),
                                                        color_mode=COLOR_MODE,
                                                        classes=None,
                                                        class_mode='categorical',
                                                        batch_size=batch_size,
                                                        shuffle=shuffle,
                                                        seed=seed,
                                                        subset='validation',
                                                        )
        else:
            train_generator = datagen.flow_from_dataframe(splash_df,
                                                        directory=data_dir,
                                                        x_col='filename',
                                                        y_col='class',
                                                        target_size=(target_width, target_height),
                                                        color_mode=COLOR_MODE,
                                                        classes=None,
                                                        class_mode='categorical',
                                                        batch_size=batch_size,
                                                        shuffle=shuffle,
                                                        seed=seed
                                                        )
            
            valid_generator = []
        
        data_generator = (train_generator, valid_generator)

    return data_generator, classes
