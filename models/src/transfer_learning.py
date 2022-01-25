import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras.models import load_model

from model_validation import TransferLearningParams, DataAugmentationParams
from helper_utils import TrainCustomCallback, data_processing


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', help='output directory')
    parser.add_argument('valid_dir', help='output directory')
    parser.add_argument('model_dir', help='output directory')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('parameters', help='list of training parameters')
    args = parser.parse_args()

    train_dir = args.train_dir
    valid_dir = args.valid_dir
    model_dir = args.model_dir
    out_dir = args.out_dir
    transfer_parameters = TransferLearningParams(**json.loads(args.parameters))
    data_parameters = DataAugmentationParams(**json.loads(args.parameters))

    print('Device: ', tf.test.gpu_device_name())
    classes = [subdir for subdir in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, subdir))]
    class_num = len(classes)

    train_generator = data_processing(data_parameters, train_dir, classes)
    if valid_dir != 'None':
        valid_generator = data_processing(data_parameters, valid_dir, classes)
    else:
        valid_generator = []

    pooling = transfer_parameters.pooling
    epochs = transfer_parameters.epochs
    start_layer = transfer_parameters.init_layer

    model = load_model(model_dir)
    model.trainable = True
    for layers in model.layers[:start_layer]:
        layers.trainable = False

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # fit model while also keeping track of data for dash plots.
    model.fit(train_generator,
              epochs=epochs,
              verbose=0,
              validation_data=valid_generator,
              callbacks=[TrainCustomCallback()],
              shuffle=True)

    # save model
    model.save(out_dir+'/model.h5')
    print("Saved to disk")
