import argparse
import json
import os

import numpy as np
from tensorflow.keras.models import load_model

from model_validation import DataAugmentationParams
from helper_utils import TestCustomCallback, data_processing


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# The following script tests the trained model given by the user, and returns the
# results as the probability per class
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', help='output directory')
    parser.add_argument('model_dir', help='input directory')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('parameters', help='list of parameters')
    args = parser.parse_args()

    test_dir = args.test_dir
    model_dir = args.model_dir
    out_dir = args.out_dir
    data_parameters = DataAugmentationParams(**json.loads(args.parameters))

    classes = [subdir for subdir in sorted(os.listdir(test_dir)) if os.path.isdir(os.path.join(test_dir, subdir))]
    class_num = len(classes)

    test_generator = data_processing(data_parameters, test_dir, classes)
    loaded_model = load_model(model_dir+'/model.h5')
    prob = loaded_model.predict(test_generator,
                                verbose=0,
                                callbacks=[TestCustomCallback()],)
    np.savetxt(out_dir + '/results.csv', prob, delimiter=' ')
