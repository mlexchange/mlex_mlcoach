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
    parser.add_argument('valid_dir', help='output directory')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('parameters', help='list of training parameters')
    args = parser.parse_args()

    train_dir = args.train_dir
    valid_dir = args.valid_dir
    out_dir = args.out_dir
    train_parameters = TrainingParams(**json.loads(args.parameters))
    data_parameters = DataAugmentationParams(**json.loads(args.parameters))

    print(tf.test.gpu_device_name())
    classes = [subdir for subdir in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, subdir))]
    class_num = len(classes)

    train_generator = data_processing(data_parameters, train_dir, classes, True)
    # check if there is a validation set
    if valid_dir != 'None':
        valid_generator = data_processing(data_parameters, valid_dir, classes, True)
    else:
        valid_generator = []

    pooling = train_parameters.pooling
    epochs = train_parameters.epochs
    nn_model = train_parameters.nn_model
    optimizer = train_parameters.optimizer
    learning_rate = train_parameters.learning_rate
    loss_func = train_parameters.loss_function

    opt = compile("tf.keras.optimizers." + optimizer + "(learning_rate=" + str(learning_rate) + ")", "<string>", "eval")
    code = compile("tf.keras.applications." + nn_model +
                   "(include_top=True, weights=None, input_tensor=None," + "pooling=" + pooling +
                   ", classes= class_num)", "<string>", "eval")
    model = eval(code)
    model.compile(optimizer=opt,        # before adam
                  loss=loss_func,       # before categorical_crossentropy
                  metrics=['accuracy'])
    model.summary()
    tf.keras.utils.plot_model(model, out_dir+'/model_layout.png', show_shapes=True)     # plot NN
    print('Length:', len(model.layers), 'layers')                                       # number of layers

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
