import argparse
import json
import os

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from model_validation import DataAugmentationParams
from helper_utils import TestCustomCallback, data_processing


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

    (test_generator, tmp) = data_processing(data_parameters, test_dir)
    try:
        test_filenames = test_generator.filenames
        test_filenames = [test_dir + '/' +  i for i in test_filenames]
        classes = [subdir for subdir in sorted(os.listdir(test_dir)) if os.path.isdir(os.path.join(test_dir, subdir))]
    except Exception as e:
        test_filenames = list(range(len(test_generator.__dict__['x'])))     # list of indexes
        test_filenames = [str(x) for x in test_filenames] 
        classes = np.unique(test_generator.__dict__['y'], axis=0)          # list of classes
        classes = [str(x) for x in classes]

    df_files = pd.DataFrame(test_filenames, columns=['filename'])
    loaded_model = load_model(model_dir)
    prob = loaded_model.predict(test_generator,
                                verbose=0,
                                callbacks=[TestCustomCallback(test_filenames, classes)])
    df_prob = pd.DataFrame(prob)
    df_results = pd.concat([df_files,df_prob], axis=1)
    df_results.to_csv(out_dir + '/results.csv', index=False)
