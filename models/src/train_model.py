import argparse
import json
import os

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
    valid_dir = args.valid_dir
    out_dir = args.out_dir
    train_parameters = TrainingParams(**json.loads(args.parameters))
    data_parameters = DataAugmentationParams(**json.loads(args.parameters))

    print(tf.test.gpu_device_name())
    train_generator = data_processing(data_parameters, train_dir)

    pooling = train_parameters.pooling
    epochs = train_parameters.epochs
    nn_model = train_parameters.nn_model
    optimizer = train_parameters.optimizer.value
    learning_rate = train_parameters.learning_rate
    loss_func = train_parameters.loss_function.value

    opt_code = compile("tf.keras.optimizers." + optimizer + "(learning_rate=" + str(learning_rate) + ")", "<string>", "eval")
    model_code = compile("tf.keras.applications." + nn_model +
                         "(include_top=True, weights=None, input_tensor=None," + "pooling=" + pooling +
                         ", classes= class_num)", "<string>", "eval")
    model = eval(model_code)
    model.compile(optimizer=eval(opt_code),         # default adam
                  loss=loss_func,                   # default categorical_crossentropy
                  metrics=['accuracy'])
    model.summary()
    tf.keras.utils.plot_model(model, out_dir+'/model_layout.png', show_shapes=True)     # plot NN
    print('Length:', len(model.layers), 'layers')                                       # number of layers

    # fit model while also keeping track of data for dash plots.
    model.fit(train_generator,
              epochs=epochs,
              verbose=0,
              callbacks=[TrainCustomCallback()],
              shuffle=data_parameters.shuffle)

    # save model
    model.save(out_dir+'/model.h5')
    print("Saved to disk")
