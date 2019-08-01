#!/usr/bin/python3

# Copyright 2019 Peter Gazdik

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import abc
import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


class ModelInterface(abc.ABC):
    def __init__(self, keras_model: keras.models.Model):
        self.model = keras_model

    @abc.abstractclassmethod
    def load(cls, model_fname):
        pass

    @abc.abstractclassmethod
    def build(cls, input_dim, out_dim, hidden_act, out_act):
        pass

    def train(self, train_generator, validation_generator, out_fname,
              epochs=1, monitor='val_loss', mode='min', min_delta=0, patience=10,
              callbacks=None):

        cbs = []
        cbs.append(ModelCheckpoint(out_fname, monitor=monitor, mode=mode,
                                   save_best_only=True, verbose=1))
        cbs.append(EarlyStopping(monitor=monitor, mode=mode,
                                 patience=patience, verbose=1,
                                 min_delta=min_delta))

        if callbacks is not None:
            cbs += callbacks

        self.model.fit_generator(train_generator,
                                 validation_data=validation_generator,
                                 epochs=epochs, callbacks=cbs,
                                 verbose=2)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, file_name):
        self.model.save(file_name)
