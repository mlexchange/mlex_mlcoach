import argparse
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from model_validation import DataAugmentationParams
from helper_utils import data_processing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', help='output directory')
    parser.add_argument('model_dir', help='input directory')
    parser.add_argument('parameters', help='list of parameters')
    args = parser.parse_args()

    test_dir = args.test_dir
    model_dir = args.model_dir
    parameters = DataAugmentationParams(**json.loads(args.parameters))

    test_generator = data_processing(parameters, test_dir)

    loaded_model = load_model(model_dir)
    results = loaded_model.evaluate(test_generator,
                                    verbose=False)
    print("test loss, test acc: " + str(results[0]) + ", " + str(results[1]), flush=True)
