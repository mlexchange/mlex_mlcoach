import argparse
import config as cfg
import json
import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.callbacks import History
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


# dimensions of our images.
img_width, img_height = 224, 224
input_shape = ((img_width, img_height))
IMG_SIZE,IMG_SIZE = 224, 224
nb_channels = 3


# Process the images and allows randomization as part of the training with
# random flips or angles if you allow it
def data_processing(values, test_data_dir):
    rotation_angle = values[0]
    horizontal_flip = 'horiz' in values[1]
    vertical_flip = 'vert' in values[1]
    batch_size = values[2]

    test_datagen = ImageDataGenerator(
        rotation_range=rotation_angle,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip)

    first_data = glob.glob(test_data_dir + '/**/*.*', recursive=True)
    data_type = os.path.splitext(first_data[1])[-1]

    if data_type == '.jpeg':

        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_height, img_width),
            color_mode="rgb",
            batch_size=1,
            class_mode='categorical',
            shuffle=False)

    elif data_type == '.npy':
        test_paths = glob.glob(test_data_dir + '/**/*.*', recursive=True)

        i = 0
        labels = {}
        for name in classes:
            labels[name] = i
            i += 1

        xTest = list()
        yTest = list()

        for path in test_paths:
            tmp_arr = np.load(path)

            input_arr = np.array([[[0, 0, 0] for y in
                                   range(len(tmp_arr))] for x in range(len(tmp_arr))])
            for x in range(len(tmp_arr)):
                for y in range(len(tmp_arr)):
                    rgb = int((255 * tmp_arr[x][y]) + 0.5)
                    input_arr[x][y] = [rgb, rgb, rgb]
            xTest.append(input_arr)
            yTest.append(labels[path.split('/')[-2]])

        xTest = np.array(xTest)
        yTest = tf.keras.utils.to_categorical(yTest, num_classes=14)

        test_generator = test_datagen.flow(
            x=xTest,
            y=yTest,
            batch_size=batch_size,
            shuffle=True)

    return test_generator


# keras callbacks for model training.  Threads while keras functions are running
# so that you can see training or evaluation of the model in progress
# class CustomCallback(tf.keras.callbacks.Callback):
#     def on_test_batch_end(self, batch, logs=None):
#         print(str(logs.get('accuracy')) + ' ' + str(logs.get('loss')) + '\n', flush=True)


# creates a model based on the options given by the user in the streamlit
# interface and trains it off of train_generator data
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', help='input directory')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('parameters', help='list of parameters')
    args = parser.parse_args()
    parameters = json.loads(args.parameters)

    cf = cfg.Config('main.cfg')
    DATA_DIR = cf['DATA_DIR']
    TRAIN_DIR = cf['TRAIN_DATA_DIR']
    VAL_DIR = cf['VALIDATION_DATA_DIR']
    TEST_DIR = cf['TEST_DATA_DIR']
    MODEL_DIR = args.in_dir

    rotation_angle = parameters['rotation_angle']
    image_flip = parameters['image_flip']
    batch_size = parameters['batch_size']

    test_generator = data_processing([rotation_angle, image_flip, batch_size], TEST_DIR)

    loaded_model = load_model(MODEL_DIR+'/my_model.h5')
    results = loaded_model.evaluate(test_generator,
                                    verbose=0)
    print("test loss, test acc: " + str(results[0]) + ", " + str(results[1]), flush=True)
