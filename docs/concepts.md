# Concepts

## Training with TF-NeuralNetworks
You can train an assortment of neural networks under different conditions according to the
definition of the following parameters:

### Data Augmentation
* Rotation Angle: Degree range for random rotations.
* Image Flip: Randomly flip inputs horizontally/vertically.

### Training setup
* Shuffle: Shuffle dataset.
* Batch Size: The number of images in a batch.
* Validation Percentage: Percentage of training images that should be used for validation.
* Pooling: Optional pooling mode.
* Optimizer: A specific implementation of the gradient descent algorithm.
* Loss Function.
* Learning Rate: A scalar used to train a model via gradient descent.
* Number of Epochs: An epoch is a full training pass over the entire dataset such that 
each image has been seen once.
* Seed: Initialization reference for the pseudo-random number generator. Set up this value 
for the reproduction of the results.

### Network Architecture
* ML Model: Definition of the network architecture, the options are:
  * [Xception](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception)
  * [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16)
  * [VGG19](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/VGG19)
  * [ResNet101](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/ResNet101)
  * [ResNet152](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/ResNet152)
  * [ResNet50V2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet50V2)
  * [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50)
  * [InceptionV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3)
  * [DenseNet201](https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet201)
  * [NASNetLarge](https://www.tensorflow.org/api_docs/python/tf/keras/applications/nasnet/NASNetLarge)
  * [InceptionResNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_resnet_v2/InceptionResNetV2)
  * [DenseNet169](https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet169)

Let the dataset dimensions be defined as (NxLxWxC), where N is the number of
images, LxW is the size of an individual image, and C is the number of channels. Please
note that this approach will resize your dataset to (Nx224x224x3) before being analyzed 
by the selected ML model to comply with the neural network architecture. For more 
information, please refer to the hyperlink of each architecture.

## Output
The output of the training step is the trained model `model.h5`.

## Prediction with TF-NeuralNetworks
To predict the labels of a given testing dataset, you can define the following 
parameters:

### Data Augmentation
* Rotation Angle: Degree range for random rotations.
* Image Flip: Randomly flip inputs horizontally/vertically.

### Testing setup
* Shuffle: Shuffle dataset. For testing purpose, this value is recommended to be False.
* Batch Size: The number of images in a batch.
* Seed: Initialization reference for the pseudo-random number generator. Set up this value 
for the reproduction of the results.

Similarly to the training step, this approach will resize your dataset to (Nx224x224x3) before being analyzed 
by the selected ML model.

## Output
The output of the prediction step is `results.csv` file with the list of filenames and their 
corresponding probabilities per class.
