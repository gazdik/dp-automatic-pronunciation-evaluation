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

import shutil
import numpy as np
import os
import keras
import random
import math


def swap(x: np.array, y: np.array, choice: np.array):
    tmp = np.empty_like(x)
    tmp[choice] = x[choice]
    x[choice] = y[choice]
    y[choice] = tmp[choice]


def scatter_batches(batch_a, batch_b, choice_percentage=1/3):
    # Generate a random choice
    length = min(len(batch_a[0]), len(batch_b[0]))
    choice_size = math.ceil(length * choice_percentage)
    choice = np.random.choice(length, size=choice_size, replace=False)

    # Swap
    swap(batch_a[0], batch_b[0], choice)
    swap(batch_a[1], batch_b[1], choice)


def scatter_multi_batches(batch_a, batch_b, choice_percentage=1/3):
    # Generate a random choice
    length = min(len(batch_a[0]), len(batch_b[0]))
    choice_size = math.ceil(length * choice_percentage)
    choice = np.random.choice(length, size=choice_size, replace=False)

    # Swap
    swap(batch_a[0], batch_b[0], choice)
    if isinstance(batch_a[1], (list, tuple)):
        for i in range(len(batch_a[1])):
            swap(batch_a[1][i], batch_b[1][i], choice)


class DataLoader:
    def __init__(self, input_it, output_dict, tmp_dir, batch_size):
        self.input_it = input_it
        self.output_dict = output_dict
        self.tmp_dir = tmp_dir
        self.batch_size = batch_size

        # Initialise data structures
        self.idx = None
        self.batch_idxs = []
        self.input_frames = None
        self.output_frames = None

        self.init_data()

    def init_data(self):
        # Load the first frame
        self.idx = 0
        key, self.input_frames = next(self.input_it)
        self.output_frames = self.output_dict[key]

    def load_data(self) -> list:
        """ Load all data and separate them into batches
        :return: indexes of the loaded batches
        """
        # Create tmp directory if it doesn't exist
        os.makedirs(self.tmp_dir, exist_ok=True)

        batch_idx = 0
        batch = self.get_next_batch()

        while batch is not None:
            self.save_batch(batch_idx, batch)

            self.batch_idxs.append(batch_idx)
            batch_idx += 1
            batch = self.get_next_batch()

        if batch_idx == 0:
            raise EOFError("generator_interface.py: No input data")

        return self.batch_idxs

    def get_next_batch(self):
        input_list = []
        output_list = []

        while len(input_list) < self.batch_size:
            input_frame, output_frame = self.get_next_frame()

            if input_frame is None:
                break

            input_list.append(input_frame)
            output_list.append(output_frame)

        if len(input_list) == 0:
            return None

        return np.vstack(input_list), np.vstack(output_list)

    def get_next_frame(self):
        """ Get next frame in input data """
        if self.idx >= len(self.input_frames):
            try:
                while True:
                    key, self.input_frames = next(self.input_it)
                    if key in self.output_dict:
                        break
            except StopIteration:
                return None, None

            self.idx = 0
            self.output_frames = self.output_dict[key]

        inputs = self.input_frames[self.idx]
        outputs = self.output_frames[self.idx]
        self.idx += 1

        return inputs, outputs

    def save_batch(self, batch_idx, batch):
        """ Save batch into a file """
        in_file_name = str(batch_idx) + '.in.npy'
        np.save(os.path.join(self.tmp_dir, in_file_name), batch[0])
        out_file_name = str(batch_idx) + '.out.npy'
        np.save(os.path.join(self.tmp_dir, out_file_name), batch[1])

    def get_batch(self, batch_idx):
        """ Returns batch on given index"""
        in_file_name = str(batch_idx) + '.in.npy'
        inputs = np.load(os.path.join(self.tmp_dir, in_file_name))
        out_file_name = str(batch_idx) + '.out.npy'
        outputs = np.load(os.path.join(self.tmp_dir, out_file_name))

        return inputs, outputs

    def input_dim(self):
        """ Returns input dimension """
        return len(self.input_frames[0])

    def output_dim(self):
        """ Returns output dimension """
        return len(self.output_frames[0])

    def scatter(self):
        """ Scatter elements between batches to increase randomness """
        idxs = self.batch_idxs.copy()
        random.shuffle(idxs)

        batch0 = self.get_batch(0)
        for i in range(0, len(idxs) - 1, 2):
            batch1 = self.get_batch(idxs[i])
            batch2 = self.get_batch(idxs[i + 1])

            scatter_batches(batch1, batch2)

            self.save_batch(idxs[i], batch1)
            self.save_batch(idxs[i + 1], batch2)

    def delete_tmp_dir(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


class MultiOutDataLoader(DataLoader):
    def __init__(self, input_it, output_dict, tmp_dir, batch_size):
        self.num_outputs = len(output_dict)

        DataLoader.__init__(self, input_it, output_dict, tmp_dir, batch_size)

    def init_data(self):
        # Load the first frame
        self.idx = 0
        key, self.input_frames = next(self.input_it)

        self.output_frames = [None] * self.num_outputs
        for i in range(self.num_outputs):
            self.output_frames[i] = self.output_dict[i][key]

    def get_next_batch(self):
        input_list = []
        output_lists = [[] for i in range(self.num_outputs)]

        while len(input_list) < self.batch_size:
            input_frame, output_frames = self.get_next_frame()

            if input_frame is None:
                break

            input_list.append(input_frame)
            for i in range(self.num_outputs):
                output_lists[i].append(output_frames[i])

        if len(input_list) == 0:
            return None

        # Convert output lists to numpy arrays
        for i in range(self.num_outputs):
            output_lists[i] = np.vstack(output_lists[i])

        return np.vstack(input_list), output_lists

    def get_next_frame(self):
        """ Get next frame in input data """
        if self.idx >= len(self.input_frames):
            try:
                key, self.input_frames = next(self.input_it)
            except StopIteration:
                return None, None

            self.idx = 0
            for i in range(self.num_outputs):
                self.output_frames[i] = self.output_dict[i][key]

        inputs = self.input_frames[self.idx]
        outputs = []
        for i in range(self.num_outputs):
            outputs.append(self.output_frames[i][self.idx])

        self.idx += 1

        return inputs, outputs

    def save_batch(self, batch_idx, batch):
        """ Save batch into a file """
        in_file_name = str(batch_idx) + '.in.npy'
        np.save(os.path.join(self.tmp_dir, in_file_name), batch[0])
        for i, output in enumerate(batch[1]):
            out_file_name = str(batch_idx) + '.out' + str(i) + '.npy'
            np.save(os.path.join(self.tmp_dir, out_file_name), output)

    def get_batch(self, batch_idx):
        """ Returns batch on given index"""
        in_file_name = str(batch_idx) + '.in.npy'
        inputs = np.load(os.path.join(self.tmp_dir, in_file_name))
        outputs = []
        for i in range(self.num_outputs):
            out_file_name = str(batch_idx) + '.out' + str(i) + '.npy'
            outputs.append(np.load(os.path.join(self.tmp_dir, out_file_name)))

        return inputs, outputs

    def output_dim(self):
        """ Returns output dimension """
        out_dims = []
        for i in range(self.num_outputs):
            out_dims.append(len(self.output_frames[i][0]))

        return out_dims

    def scatter(self):
        """ Scatter elements between batches to increase randomness """
        idxs = self.batch_idxs.copy()
        random.shuffle(idxs)

        for i in range(0, len(idxs) - 1, 2):
            batch1 = self.get_batch(idxs[i])
            batch2 = self.get_batch(idxs[i + 1])

            # Swap input elements
            scatter_multi_batches(batch1, batch2)

            self.save_batch(idxs[i], batch1)
            self.save_batch(idxs[i + 1], batch2)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_loader: DataLoader, shuffle=True):
        self.data_loader = data_loader
        self.shuffle = shuffle

        self.idxs = self.data_loader.load_data()

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        idx = self.idxs[item]
        return self.data_loader.get_batch(idx)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.idxs)
            self.data_loader.scatter()

    def input_dim(self):
        return self.data_loader.input_dim()

    def output_dim(self):
        return self.data_loader.output_dim()

    def delete_tmp_dir(self):
        self.data_loader.delete_tmp_dir()





