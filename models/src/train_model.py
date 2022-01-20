import argparse
import config as cfg
import json
import glob
from PIL import Image
import os

from keras.callbacks import History
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_validation import DataAugmentationParams, TrainingParams


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Process the images and allows randomization as part of the training with
# random flips or angles if you allow it
def data_processing(values, train_data_dir, val_data_dir):
    rotation_angle = values[0]
    horizontal_flip = 'horiz' in values[1]
    vertical_flip = 'vert' in values[1]
    batch_size = values[2]
    target_size = values[3]

    train_datagen = ImageDataGenerator(
        rotation_range=rotation_angle,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip)

    first_data = glob.glob(train_data_dir + '/**/*.*', recursive=True)
    data_type = os.path.splitext(first_data[0])[-1]

    im = Image.open(first_data[0])
    if im.mode == 'RGB':
        color_mode = 'rgb'
    if im.mode == 'RGBA':
        color_mode = 'rgba'
    if im.mode == 'L':
        color_mode = 'grayscale'

    if data_type == '.jpeg':
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=target_size,
            batch_size=batch_size,
            color_mode=color_mode,
            class_mode='categorical')

        valid_generator = []

    elif data_type == '.npy':
        train_paths = first_data
        validation_paths = glob.glob(val_data_dir + '/**/*.*', recursive=True)

        i = 0
        labels = {}
        for name in classes:
            labels[name] = i
            i += 1

        xTrain = list()
        yTrain = list()
        xValid = list()
        yValid = list()

        for path in train_paths:
            tmp_arr = np.load(path)

            input_arr = np.array([[[0, 0, 0] for y in
                                   range(len(tmp_arr))] for x in range(len(tmp_arr))])
            for x in range(len(tmp_arr)):
                for y in range(len(tmp_arr)):
                    rgb = int((255 * tmp_arr[x][y]) + 0.5)
                    input_arr[x][y] = [rgb, rgb, rgb]
            xTrain.append(input_arr)
            yTrain.append(labels[path.split('/')[-2]])

        for path in validation_paths:
            tmp_arr = np.load(path)

            input_arr = np.array([[[0, 0, 0] for y in
                                   range(len(tmp_arr))] for x in range(len(tmp_arr))])
            for x in range(len(tmp_arr)):
                for y in range(len(tmp_arr)):
                    rgb = int((255 * tmp_arr[x][y]) + 0.5)
                    input_arr[x][y] = [rgb, rgb, rgb]
            xValid.append(input_arr)
            yValid.append(labels[path.split('/')[-2]])

        xTrain = np.array(xTrain)
        xValid = np.array(xValid)
        yTrain = tf.keras.utils.to_categorical(yTrain, num_classes=14)
        yValid = tf.keras.utils.to_categorical(yValid, num_classes=14)

        train_generator = train_datagen.flow(
            x=xTrain,
            y=yTrain,
            batch_size=batch_size,
            shuffle=True)

        valid_generator = test_datagen.flow(
            x=xValid,
            y=yValid,
            batch_size=batch_size,
            shuffle=True)

    return train_generator, valid_generator


# keras callbacks for model training.  Threads while keras functions are running
# so that you can see training or evaluation of the model in progress
class CustomCallback(tf.keras.callbacks.Callback):
    # For model training
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            print('epoch loss\n', flush=True)
        print(str(epoch)+' '+str(logs.get('loss'))+'\n', flush=True)

    def on_train_end(self, logs=None):
        print('Train process completed', flush=True)


# creates a model based on the options given by the user in the streamlit
# interface and trains it off of train_generator data
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('parameters', help='list of training parameters')
    args = parser.parse_args()
    train_parameters = TrainingParams(**json.loads(args.parameters))
    data_aug_parameters = train_parameters.data_augmentation

    cf = cfg.Config('main.cfg')
    DATA_DIR = cf['DATA_DIR']
    TRAIN_DIR = cf['TRAIN_DATA_DIR']
    VAL_DIR = cf['VALIDATION_DATA_DIR']
    MODEL_DIR = args.out_dir

    rotation_angle = data_aug_parameters.rotation_angle
    image_flip = data_aug_parameters.image_flip
    batch_size = data_aug_parameters.batch_size
    target_size = data_aug_parameters.target_size

    print('Device: ', tf.test.gpu_device_name())
    train_generator, valid_generator = \
        data_processing([rotation_angle, image_flip, batch_size, target_size], TRAIN_DIR, VAL_DIR)

    CLASSES = [subdir for subdir in sorted(os.listdir(TRAIN_DIR)) if
               os.path.isdir(os.path.join(TRAIN_DIR, subdir))]
    class_num = len(CLASSES)

    pooling = train_parameters.pooling
    stepoch = train_parameters.stepoch
    epochs = train_parameters.epochs
    nn_model = train_parameters.nn_model

    code = compile("tf.keras.applications." + nn_model +
                   "(include_top=True, weights=None, input_tensor=None," + "pooling=" + pooling +
                   ", classes= class_num)", "<string>", "eval")
    model = eval(code)

    # tf.keras.utils.plot_model(model, "model_layout.png", show_shapes=True)
    model.compile(
        optimizer='adam', loss='categorical_crossentropy',
        metrics=['accuracy'])

    # fit model while also keeping track of data for dash plots.
    model.fit(train_generator,
              steps_per_epoch=stepoch,
              epochs=epochs,
              verbose=0,
              validation_data=valid_generator,
              callbacks=[CustomCallback()],
              shuffle=True)

    # save model
    model.save(MODEL_DIR+'/my_model.h5')
    print("Saved to disk")
