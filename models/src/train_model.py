import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from model_validation import TrainingParams, DataAugmentationParams
from helper_utils import TrainCustomCallback, data_processing


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', help='output directory')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('parameters', help='list of training parameters')
    args = parser.parse_args()

    train_dir = args.train_dir
    out_dir = args.out_dir
    train_parameters = TrainingParams(**json.loads(args.parameters))
    data_parameters = DataAugmentationParams(**json.loads(args.parameters))
    print(tf.test.gpu_device_name())
    (train_generator, valid_generator), classes = data_processing(data_parameters, train_dir)
    try:
        train_filenames = train_generator.filenames
    except Exception as e:
        train_filenames = list(range(len(train_generator.__dict__['x'])))     # list of indexes
    class_num = len(classes)

    weights = train_parameters.weights
    epochs = train_parameters.epochs
    nn_model = train_parameters.nn_model
    optimizer = train_parameters.optimizer.value
    learning_rate = train_parameters.learning_rate
    loss_func = train_parameters.loss_function.value

    opt_code = compile(f'tf.keras.optimizers.{optimizer}(learning_rate={learning_rate})', '<string>', 'eval')
    model_code = compile(f'tf.keras.applications.{nn_model}(include_top=True, weights={weights}, input_tensor=None, classes= class_num)',
                          '<string>', 'eval')
    model = eval(model_code)
    model.compile(optimizer=eval(opt_code),         # default adam
                  loss=loss_func,                   # default categorical_crossentropy
                  metrics=['accuracy'])
    # model.summary()
    # tf.keras.utils.plot_model(model, out_dir+'/model_layout.png', show_shapes=True)       # plot NN
    print('Length:', len(model.layers), 'layers')                                           # number of layers

    # fit model while also keeping track of data for dash plots.
    model.fit(train_generator,
              validation_data=valid_generator,
              epochs=epochs,
              verbose=0,
              callbacks=[TrainCustomCallback()],
              shuffle=data_parameters.shuffle)

    # save model
    model.save(out_dir+'/model.h5')
    df_classes = pd.DataFrame(classes)
    df_classes.to_csv(out_dir + '/classes.csv', index=False)
    print("Saved to disk")
