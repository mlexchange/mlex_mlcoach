from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class Pooling(str, Enum):
    none = 'None'
    maximum = 'Maximum'
    average = 'Average'


class NNModel(str, Enum):
    xception = 'Xception'
    vgg16 = 'VGG16'
    vgg19 = 'VGG19'
    resnet101 = 'ResNet101'
    resnet152 = 'ResNet152'
    resnet50v2 = 'ResNet50V2'
    resnet50 = 'ResNet50'
    resnet152v2 = 'ResNet152V2'
    inceptionv3 = 'InceptionV3'
    densenet201 = 'DenseNet201'
    nasnetlarge = 'NASNetLarge'
    inceptionresnetv2 = 'InceptionResNetV2'
    densenet169 = 'DenseNet169'


class ImageFlip(str, Enum):
    none = 'None'
    vert = 'Vertical'
    horiz = 'Horizontal'
    both = 'Both'


class DataAugmentationParams(BaseModel):
    rotation_angle: int = Field(description='rotation angle')
    image_flip: ImageFlip
    batch_size: int = Field(description='batch size')


class TrainingParams(DataAugmentationParams):
    pooling: Pooling
    epochs: int = Field(description='number of epochs')
    nn_model: NNModel


class TransferLearningParams(TrainingParams):
    init_layer: int = Field(description='initial layer')
    nn_model: Optional[NNModel]
