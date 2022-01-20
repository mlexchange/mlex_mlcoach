import urllib.request

import dash
import json
import glob
import os
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from kwarg_editor import JSONParameterEditor


# dimensions of our images.
img_width, img_height = 223, 223
input_shape = ((img_width, img_height))
IMG_SIZE,IMG_SIZE = 223, 223
nb_channels = 3


class SimpleJob:
    def __init__(self,
                 user,
                 job_type,
                 description,
                 deploy_location,
                 gpu,
                 data_uri,
                 container_uri,
                 container_cmd,
                 container_kwargs,
                 mlex_app = 'mlcoach'):
        self.user = user
        self.mlex_app = mlex_app
        self.job_type = job_type
        self.description = description
        self.deploy_location = deploy_location
        self.gpu = gpu
        self.data_uri = data_uri
        self.container_uri = container_uri
        self.container_cmd = container_cmd
        self.container_kwargs = container_kwargs

    def launch_job(self):
        """
        Send job to computing service
        :return:
        """
        url = 'http://job-service:8080/api/v0/jobs'
        return requests.post(url, json=self.__dict__).status_code


# Queries the job from the computing database
def get_job(user, mlex_app, job_type=None, deploy_location=None):
    url = 'http://job-service:8080/api/v0/jobs?'
    if user:
        url += ('&user=' + user)
    if mlex_app:
        url += ('&mlex_app=' + mlex_app)
    if job_type:
        url += ('&job_type=' + job_type)
    if deploy_location:
        url += ('&deploy_location=' + deploy_location)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data


# Generates the dash widgets based on the json file associated with the app
def generate_dash_widget(dash_schema):
    parameters_schema = dash_schema['gui_parameters']
    parameters = JSONParameterEditor(_id={'type': 'labelmaker'},
                                     parameters=parameters_schema)
    return parameters


# Process the images and allows randomization as part of the training with
# random flips or angles if you allow it
def data_processing(values, train_data_dir, val_data_dir, test_data_dir):
    rotation_angle = values[0]
    horizontal_flip = 'horiz' in values[1]
    vertical_flip = 'vert' in values[1]
    batch_size = values[2]
    train_datagen = ImageDataGenerator(
        rotation_range=rotation_angle,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip)

    test_datagen = ImageDataGenerator(
        rotation_range=rotation_angle,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip)

    first_data = glob.glob(train_data_dir + '/**/*.*', recursive=True)
    data_type = os.path.splitext(first_data[1])[-1]

    if data_type == '.jpeg':
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode="rgb",
            class_mode='categorical')

        valid_generator = []

        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_height, img_width),
            color_mode="rgb",
            batch_size=1,
            class_mode='categorical',
            shuffle=False)

    elif data_type == '.npy':
        train_paths = first_data
        validation_paths = glob.glob(val_data_dir + '/**/*.*', recursive=True)
        test_paths = glob.glob(test_data_dir + '/**/*.*', recursive=True)

        i = 0
        labels = {}
        for name in classes:
            labels[name] = i
            i += 1

        xTrain = list()
        yTrain = list()
        xValid = list()
        yValid = list()
        xTest = list()
        yTest = list()

        for path in train_paths:
            tmp_arr = np.load(path)

            input_arr = np.array([[[0, 0, 0] for y in
                                   range(len(tmp_arr))] for x in range(len(tmp_arr))])
            for x in range(len(tmp_arr)):
                for y in range(len(tmp_arr)):
                    rgb = int((255 * tmp_arr[x][y]) + 0.5)
                    input_arr[x][y] = [rgb, rgb, rgb]
            xTrain.append(input_arr)
            yTrain.append(labels[path.split('/')[-2]])

        for path in validation_paths:
            tmp_arr = np.load(path)

            input_arr = np.array([[[0, 0, 0] for y in
                                   range(len(tmp_arr))] for x in range(len(tmp_arr))])
            for x in range(len(tmp_arr)):
                for y in range(len(tmp_arr)):
                    rgb = int((255 * tmp_arr[x][y]) + 0.5)
                    input_arr[x][y] = [rgb, rgb, rgb]
            xValid.append(input_arr)
            yValid.append(labels[path.split('/')[-2]])

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

        xTrain = np.array(xTrain)
        xValid = np.array(xValid)
        xTest = np.array(xTest)
        yTrain = tf.keras.utils.to_categorical(yTrain, num_classes=14)
        yValid = tf.keras.utils.to_categorical(yValid, num_classes=14)
        yTest = tf.keras.utils.to_categorical(yTest, num_classes=14)

        train_generator = train_datagen.flow(
            x=xTrain,
            y=yTrain,
            batch_size=batch_size,
            shuffle=True)

        valid_generator = test_datagen.flow(
            x=xValid,
            y=yValid,
            batch_size=batch_size,
            shuffle=True)

        test_generator = test_datagen.flow(
            x=xTest,
            y=yTest,
            batch_size=batch_size,
            shuffle=True)

    return train_generator, valid_generator, test_generator


# keras callbacks for model training.  Threads while keras functions are running
# so that you can see training or evaluation of the model in progress
class fit_myCallback(tf.keras.callbacks.Callback):
    # For model training
    def on_batch_end(self, epoch, logs={}):
        accuracy_chart.add_rows([logs.get('accuracy')])
        loss_chart.add_rows([logs.get('loss')])

    def on_train_end(self, logs={}):
        st.text("Accuracy: " + str(logs.get('accuracy')) + " Loss: " +
                str(logs.get('loss')))


# creates a model based on the options given by the user in the streamlit
# interface and trains it off of train_generator data
def create_model(values, train_generator, valid_generator, class_num):
    pooling = values[0]
    stepoch = values[1]
    epochs = values[2]
    value = values[3]
    print(values)
    fit_callbacks = fit_myCallback()
    code = compile(
            "tf.keras.applications." + value +
            "(include_top=True, weights=None, input_tensor=None," +
            "pooling=" + pooling +
            ", classes= class_num)", "<string>", "eval")
    model = eval(code)
    tf.keras.utils.plot_model(model, "model_layout.png", show_shapes=True)
    model.compile(
            optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])
    # fit model while also keeping track of data for dash plots.
    model.fit(train_generator,
              steps_per_epoch=stepoch,
              epochs=epochs,
              verbose=1,
              validation_data=valid_generator,
              callbacks=[fit_callbacks],
              shuffle=True)
    return model


def get_class_prob(log, start, slider_value, classes):
    end = log.find('Prediction process completed')
    if end == -1:
        end = len(log)
    log = log[start:end]
    df = pd.read_csv(StringIO(log.replace('\n\n', '\n')), sep=' ')
    try:
        res = df.iloc[slider_value]
        return 'Class: '+ classes[int(res['class'])] + '\nProbability: ' + str(res['probability'])
    except Exception as err:
        return ''


# Generate loss plot
def generate_figure(log, start):
    end = log.find('Train process completed')
    if end == -1:
        end = len(log)
    log = log[start:end]
    df = pd.read_csv(StringIO(log.replace('\n\n', '\n')), sep=' ')
    df.set_index('epoch', inplace=True)
    try:
        fig = px.line(df, markers=True)
        fig.update_layout(xaxis_title="epoch", yaxis_title="loss", margin=dict(l=20, r=20, t=20, b=20))
        return fig
    except Exception as e:
        print(e)
        return go.Figure(go.Scatter(x=[], y=[]))


# saves model as an .h5 file on local disk
def save_model(model, save_path='my_model.h5'):
    # save model
    model.save(save_path)
    print("Saved to disk")
